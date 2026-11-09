#include "picanative/shadergen.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <vector>

// The program is cut into basic blocks at every address control can reach
// other than by falling through, and runs as a loop dispatching on the
// program counter, with the same stack of call, if and loop blocks as the
// hardware: a block ends when the program counter reaches its end, then
// returns, or goes round the loop again. This is exact for any control flow,
// jumps included; the structured translation of the optimisation phase is
// checked against it.
namespace picanative::shadergen {
namespace {

enum Op : uint32_t {
  Add = 0x00, Dp3 = 0x01, Dp4 = 0x02, Dph = 0x03, Dst = 0x04, Ex2 = 0x05, Lg2 = 0x06, LitP = 0x07,
  Mul = 0x08, Sge = 0x09, Slt = 0x0A, Flr = 0x0B, Max = 0x0C, Min = 0x0D, Rcp = 0x0E, Rsq = 0x0F,
  Mova = 0x12, Mov = 0x13, DphI = 0x18, DstI = 0x19, SgeI = 0x1A, SltI = 0x1B, Break = 0x20,
  Nop = 0x21, End = 0x22, BreakC = 0x23, Call = 0x24, CallC = 0x25, CallU = 0x26, IfU = 0x27,
  IfC = 0x28, Loop = 0x29, Emit = 0x2A, SetEmit = 0x2B, JmpC = 0x2C, JmpU = 0x2D,
};

uint32_t opcode(uint32_t word) {
  const uint32_t raw = word >> 26;
  if (raw >= 0x38) return 0x38; // mad
  if (raw >= 0x30) return 0x30; // madi
  if (raw == 0x2F) return 0x2E; // cmp
  return raw;
}
constexpr uint32_t kMad = 0x38, kMadI = 0x30, kCmp = 0x2E;

bool inverted(uint32_t op) { return op == DphI || op == DstI || op == SgeI || op == SltI; }
bool flow(uint32_t op) {
  return op == Break || op == End || op == BreakC || op == Call || op == CallC || op == CallU || op == IfU ||
         op == IfC || op == Loop || op == JmpC || op == JmpU;
}
uint32_t target(uint32_t word) { return word >> 10 & 0xFFF; }
uint32_t count(uint32_t word) { return word & 0xFF; }

// An operand as a WGSL expression: the register, an address register
// applied to a uniform, the swizzle and the negation of its descriptor.
std::string operand(uint32_t reg, uint32_t index, uint32_t descriptor, int source) {
  static const int shifts[3] = {5, 14, 23};
  const int shift = shifts[source];
  const uint32_t pattern = descriptor >> shift & 0xFF;
  const bool negate = descriptor >> (shift - 1) & 1;
  std::string base;
  if (reg < 0x10) {
    base = "v[" + std::to_string(reg) + "]";
  } else if (reg < 0x20) {
    base = "r[" + std::to_string(reg - 0x10) + "]";
  } else if (index == 0) {
    base = "cu(" + std::to_string(reg - 0x20) + ")";
  } else {
    static const char* address[3] = {"a.x", "a.y", "a.z"};
    base = "cu(" + std::to_string(reg - 0x20) + " + " + address[(index - 1) % 3] + ")";
  }
  std::string swizzle = ".";
  for (int c = 0; c < 4; ++c) swizzle += "xyzw"[pattern >> (6 - c * 2) & 3];
  const std::string value = base + swizzle;
  return negate ? "(-" + value + ")" : value;
}

std::string destination(uint32_t reg) {
  if (reg < 0x10) return "o[" + std::to_string(reg) + "]";
  if (reg < 0x20) return "r[" + std::to_string(reg - 0x10) + "]";
  return "";
}

// assigns the components of `t` the destination mask selects (x is bit 3)
std::string write_masked(uint32_t reg, uint32_t mask) {
  const std::string to = destination(reg);
  if (to.empty() || (mask & 0xF) == 0) return "";
  if ((mask & 0xF) == 0xF) return to + " = t; ";
  std::string out;
  for (int c = 0; c < 4; ++c)
    if (mask & (8 >> c)) out += to + "." + "xyzw"[c] + " = t." + "xyzw"[c] + "; ";
  return out;
}

std::string condition(uint32_t word) {
  const std::string x = std::string("(cc.x == ") + (word >> 25 & 1 ? "true" : "false") + ")";
  const std::string y = std::string("(cc.y == ") + (word >> 24 & 1 ? "true" : "false") + ")";
  switch (word >> 22 & 3) {
  case 0: return "(" + x + " || " + y + ")";
  case 1: return "(" + x + " && " + y + ")";
  case 2: return x;
  default: return y;
  }
}

std::string bool_set(uint32_t word) {
  return "((U.bools.x >> " + std::to_string(word >> 22 & 0xF) + "u) & 1u) != 0u";
}

std::string compare(uint32_t mode, const std::string& a, const std::string& b) {
  static const char* ops[6] = {"==", "!=", "<", "<=", ">", ">="};
  if (mode >= 6) return "true";
  return "(" + a + " " + ops[mode] + " " + b + ")";
}

// one arithmetic instruction
std::string arithmetic(uint32_t word, const std::array<uint32_t, pica::kDescriptorSize>& descriptors) {
  const uint32_t op = opcode(word);
  if (op == kMad || op == kMadI) {
    const uint32_t d = descriptors[(word & 0x1F) % pica::kDescriptorSize];
    const uint32_t src1 = word >> 17 & 0x1F, index = word >> 22 & 3;
    uint32_t src2, src3, index2 = 0, index3 = 0;
    if (op == kMadI) {
      src2 = word >> 12 & 0x1F;
      src3 = word >> 5 & 0x7F;
      index3 = index;
    } else {
      src2 = word >> 10 & 0x7F;
      src3 = word >> 5 & 0x1F;
      index2 = index;
    }
    return "{ let t = pmul(" + operand(src1, 0, d, 0) + ", " + operand(src2, index2, d, 1) + ") + " +
           operand(src3, index3, d, 2) + "; " + write_masked(word >> 24 & 0x1F, d) + "}";
  }
  const uint32_t d = descriptors[(word & 0x7F) % pica::kDescriptorSize];
  uint32_t src1, src2, index1 = 0, index2 = 0;
  const uint32_t index = word >> 19 & 3;
  if (inverted(op)) {
    src1 = word >> 14 & 0x1F;
    src2 = word >> 7 & 0x7F;
    index2 = index;
  } else {
    src1 = word >> 12 & 0x7F;
    src2 = word >> 7 & 0x1F;
    index1 = index;
  }
  const std::string a = operand(src1, index1, d, 0), b = operand(src2, index2, d, 1);
  std::string value;
  switch (op) {
  case Add: value = a + " + " + b; break;
  case Mul: value = "pmul(" + a + ", " + b + ")"; break;
  // NaN as the hardware has it: max(0, NaN) is NaN, max(NaN, 0) is 0
  case Max: value = "pmax(" + a + ", " + b + ")"; break;
  case Min: value = "pmin(" + a + ", " + b + ")"; break;
  case Dp3: value = "vec4f(pdot3(" + a + ", " + b + "))"; break;
  case Dp4: value = "vec4f(pdot4(" + a + ", " + b + "))"; break;
  case Dph:
  case DphI: value = "vec4f(pdot3(" + a + ", " + b + ") + (" + b + ").w)"; break;
  case Mov: value = a; break;
  case Flr: value = "floor(" + a + ")"; break;
  case Rcp: value = "vec4f(1.0 / (" + a + ").x)"; break;
  case Rsq: value = "vec4f(1.0 / sqrt((" + a + ").x))"; break;
  case Ex2: value = "vec4f(exp2((" + a + ").x))"; break;
  case Lg2: value = "vec4f(log2((" + a + ").x))"; break;
  case Sge:
  case SgeI: value = "select(vec4f(0.0), vec4f(1.0), " + a + " >= " + b + ")"; break;
  case Slt:
  case SltI: value = "select(vec4f(0.0), vec4f(1.0), " + a + " < " + b + ")"; break;
  case Dst:
  case DstI: value = "vec4f(1.0, pmul1((" + a + ").y, (" + b + ").y), (" + a + ").z, (" + b + ").w)"; break;
  case LitP:
    return "{ let s = " + a + "; cc = vec2<bool>(s.x >= 0.0, s.w >= 0.0); let t = vec4f(pmax1(s.x, 0.0), " +
           "clamp(s.y, -127.9961, 127.9961), 0.0, pmax1(s.w, 0.0)); " + write_masked(word >> 21 & 0x1F, d) + "}";
  case Mova: {
    std::string out = "{ let s = " + a + "; ";
    if (d & 8) out += "a.x = i32(s.x); ";
    if (d & 4) out += "a.y = i32(s.y); ";
    return out + "}";
  }
  case kCmp: {
    const uint32_t mx = word >> 24 & 7, my = word >> 21 & 7;
    return "{ let p = " + a + "; let q = " + b + "; cc = vec2<bool>(" + compare(mx, "p.x", "q.x") + ", " +
           compare(my, "p.y", "q.y") + "); }";
  }
  default: return ""; // unknown, emit, setemit: nothing in a vertex shader
  }
  return "{ let t = " + value + "; " + write_masked(word >> 21 & 0x1F, d) + "}";
}

} // namespace

uint64_t program_key(const pica::ShaderUnit& unit) {
  uint64_t hash = 0xCBF29CE484222325ull;
  auto add = [&hash](uint32_t word) {
    for (int b = 0; b < 4; ++b) hash = (hash ^ (word >> (8 * b) & 0xFF)) * 0x100000001B3ull;
  };
  for (uint32_t word : unit.program) add(word);
  for (uint32_t word : unit.descriptors) add(word);
  add(unit.entry_point);
  return hash;
}

std::string vertex_function(const pica::ShaderUnit& unit, const std::string& uniforms) {
  const auto& program = unit.program;
  constexpr uint32_t size = pica::kProgramSize;

  // the addresses control can arrive at other than by falling through
  std::set<uint32_t> starts;
  std::set<uint32_t> visited;
  std::vector<uint32_t> work{unit.entry_point};
  auto boundary = [&](uint32_t address) {
    if (address < size && starts.insert(address).second) work.push_back(address);
  };
  // walk: every block start reached, splitting where a flow instruction is
  // or a known boundary falls; boundaries found later split blocks again in
  // the final pass, the walk only has to find them all
  starts.insert(unit.entry_point);
  while (!work.empty()) {
    const uint32_t start = work.back();
    work.pop_back();
    if (!visited.insert(start).second) continue;
    for (uint32_t pc = start; pc < size; ++pc) {
      const uint32_t word = program[pc], op = opcode(word);
      if (!flow(op)) continue;
      const uint32_t dest = target(word), n = count(word);
      switch (op) {
      case End: break;
      case Call: case CallC: case CallU:
        boundary(dest); boundary(dest + n); boundary(pc + 1);
        break;
      case JmpC: case JmpU:
        boundary(dest); boundary(pc + 1);
        break;
      case IfC: case IfU: {
        const uint32_t then_end = pc + 1 + (dest - std::min(pc + 1, dest));
        boundary(pc + 1); boundary(then_end); boundary(dest); boundary(dest + n);
        break;
      }
      case Loop:
        boundary(pc + 1); boundary(dest + 1);
        break;
      case Break: case BreakC:
        boundary(pc + 1);
        break;
      }
      break; // a flow instruction ends the walk from this start
    }
  }

  std::ostringstream w;
  w << "fn cu(i: i32) -> vec4f { if (i < 0 || i >= 96) { return vec4f(0.0); } return " << uniforms
    << ".c[i]; }\n";
  w << "fn pmul(a: vec4f, b: vec4f) -> vec4f { let p = a * b; return select(p, vec4f(0.0), (p != p) & (a == a) & (b == b)); }\n";
  w << "fn pmul1(a: f32, b: f32) -> f32 { let p = a * b; return select(p, 0.0, (p != p) & (a == a) & (b == b)); }\n";
  w << "fn pdot3(a: vec4f, b: vec4f) -> f32 { let p = pmul(a, b); return (p.x + p.y) + p.z; }\n";
  w << "fn pdot4(a: vec4f, b: vec4f) -> f32 { let p = pmul(a, b); return ((p.x + p.y) + p.z) + p.w; }\n";
  w << "fn pmax(a: vec4f, b: vec4f) -> vec4f { return select(b, a, a > b); }\n";
  w << "fn pmin(a: vec4f, b: vec4f) -> vec4f { return select(b, a, a < b); }\n";
  w << "fn pmax1(a: f32, b: f32) -> f32 { return select(b, a, a > b); }\n";
  w << "fn pica_vs(vin: array<vec4f, 16>) -> array<vec4f, 16> {\n";
  w << "  let U = " << uniforms << ";\n";
  w << "  var v = vin;\n  var o: array<vec4f, 16>;\n  var r: array<vec4f, 16>;\n";
  w << "  for (var k = 0; k < 16; k++) { r[k] = vec4f(0.0, 0.0, 0.0, 1.0); }\n";
  w << "  var a = vec3<i32>(0);\n  var cc = vec2<bool>(false, false);\n";
  // the block stack: end, return address, repeats left, increment, start, is a loop
  w << "  var be: array<u32, 32>; var br: array<u32, 32>; var bn: array<u32, 32>; var bi: array<i32, 32>;\n";
  w << "  var bs: array<u32, 32>; var bl: array<bool, 32>; var sp = 0;\n";
  w << "  var pc = " << unit.entry_point << "u;\n  var budget = 0u;\n";
  w << "  loop {\n";
  w << "    loop {\n      if (sp == 0 || pc != be[sp - 1]) { break; }\n";
  w << "      a.z += bi[sp - 1];\n";
  w << "      if (bn[sp - 1] == 0u) { pc = br[sp - 1]; sp--; } else { bn[sp - 1]--; pc = bs[sp - 1]; }\n    }\n";
  w << "    budget++;\n    if (pc >= " << size << "u || budget > 65536u) { break; }\n";
  w << "    var done = false;\n";
  w << "    switch pc {\n";

  auto push = [](const std::string& end, const std::string& ret, const std::string& repeat, const std::string& inc,
                 const std::string& start, bool loop) {
    return "if (sp < 32) { be[sp] = " + end + "; br[sp] = " + ret + "; bn[sp] = " + repeat + "; bi[sp] = " + inc +
           "; bs[sp] = " + start + "; bl[sp] = " + (loop ? "true" : "false") + "; sp++; } ";
  };
  for (uint32_t start : starts) {
    if (!visited.count(start)) continue;
    w << "      case " << start << "u: {\n";
    uint32_t pc = start;
    bool ended = false;
    for (; pc < size; ++pc) {
      if (pc != start && starts.count(pc)) break; // the next block starts here
      const uint32_t word = program[pc], op = opcode(word);
      if (!flow(op)) {
        const std::string code = arithmetic(word, unit.descriptors);
        if (!code.empty()) w << "        " << code << "\n";
        continue;
      }
      const uint32_t dest = target(word), n = count(word);
      const std::string next = std::to_string(pc + 1) + "u";
      std::string code;
      switch (op) {
      case End: code = "done = true;"; break;
      case Nop: break;
      case Call: case CallC: case CallU: {
        const std::string cond = op == Call ? "true" : op == CallU ? bool_set(word) : condition(word);
        code = "if (" + cond + ") { " +
               push(std::to_string(dest + n) + "u", next, "0u", "0", std::to_string(dest) + "u", false) + "pc = " +
               std::to_string(dest) + "u; } else { pc = " + next + "; }";
        break;
      }
      case JmpC: case JmpU: {
        // jmpu tests either value of its bool: bit 0 of the count inverts it
        const std::string cond = op == JmpC ? condition(word)
                                            : "((" + bool_set(word) + ") == " + (n & 1 ? "false" : "true") + ")";
        code = "if (" + cond + ") { pc = " + std::to_string(dest) + "u; } else { pc = " + next + "; }";
        break;
      }
      case IfC: case IfU: {
        const std::string cond = op == IfU ? bool_set(word) : condition(word);
        const uint32_t then_count = dest - std::min(pc + 1, dest);
        const std::string after = std::to_string(dest + n) + "u";
        code = "if (" + cond + ") { " +
               push(std::to_string(pc + 1 + then_count) + "u", after, "0u", "0", next, false) + "pc = " + next +
               "; } else { " + push(std::to_string(dest + n) + "u", after, "0u", "0", std::to_string(dest) + "u", false) +
               "pc = " + std::to_string(dest) + "u; }";
        break;
      }
      case Loop: {
        const std::string i = "U.ints[" + std::to_string(word >> 22 & 3) + "]";
        code = "a.z = i32((" + i + " >> 8u) & 0xFFu); " +
               push(std::to_string(dest + 1) + "u", std::to_string(dest + 1) + "u", "(" + i + " & 0xFFu)",
                    "(i32(" + i + " << 8u) >> 24)", next, true) +
               "pc = " + next + ";";
        break;
      }
      case Break: case BreakC: {
        const std::string cond = op == Break ? "true" : condition(word);
        code = "if (" + cond + ") { var to = " + next +
               "; loop { if (sp == 0) { break; } sp--; if (bl[sp]) { to = br[sp]; break; } } pc = to; } else { pc = " +
               next + "; }";
        break;
      }
      }
      if (op == Nop) continue;
      w << "        " << code << "\n";
      ended = true;
      ++pc;
      break;
    }
    if (!ended) w << "        pc = " << pc << "u;\n";
    w << "      }\n";
  }
  w << "      default: { done = true; }\n";
  w << "    }\n    if (done) { break; }\n  }\n  return o;\n}\n";
  // the flow instructions name the uniforms U.ints and U.bools
  std::string out = w.str();
  for (size_t at = 0; (at = out.find("U.", at)) != std::string::npos; at += uniforms.size() + 1)
    if (at == 0 || !(std::isalnum(static_cast<unsigned char>(out[at - 1])) || out[at - 1] == '_'))
      out.replace(at, 1, uniforms);
  return out;
}

} // namespace picanative::shadergen
