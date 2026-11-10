#include "picanative/draw.hpp"

#include "picanative/shadergen.hpp"

#include <cmath>
#include <sstream>

// Bindings of a draw pipeline (group 0):
//   0: PicaVsUniforms (uniform, dynamic offset)   1: Draw (uniform, dynamic offset)
//   2: arena, the draw's vertex arrays and indices as copied from memory (storage, array<u32>)
//   3: lookup tables, vec2 entries: lighting (24 x 256) and fog (128) (storage)
//   4-6: texture units 0-2 (texture_2d<f32>, sampled by hand with textureLoad)
//
// The vertex stage pulls its vertices from the arena: the topology and the
// index buffer decide which vertex each corner is, the vertex layout where
// its attributes are. The PICA's clip volume (-w <= z <= 0) becomes the
// GPU's (0 <= z' <= w) with z' = z + w, and the viewport is applied in clip
// space so the GPU's viewport is the whole target and the scissor the PICA's
// viewport. The depth is written by the fragment stage, from z/w
// interpolated linearly as the PICA does, rounded to 24 bits.
namespace picanative::draw {
namespace {

std::string u(uint32_t value) { return std::to_string(value) + "u"; }
std::string f(float value) {
  std::ostringstream s;
  s.precision(9);
  s << value;
  std::string out = s.str();
  if (out.find_first_of(".e") == std::string::npos) out += ".0";
  return out;
}

// where each varying lives in the vertex shader's packed outputs
struct Slot {
  int reg = -1;      // output register, -1 missing, -2 zero
  int component = 0;
};

struct Semantics {
  Slot slots[24]; // position x-w, quaternion 4-7, color 8-11, uv0 12-13, uv1 14-15, uv0 w 16, view 18-20, uv2 22-23
};

Semantics semantics(const Config& c) {
  Semantics s;
  // the output registers enabled, packed in order
  std::vector<int> enabled;
  for (int r = 0; r < 16; ++r)
    if ((c.output_mask & 0xFFFF) == 0 || c.output_mask >> r & 1) enabled.push_back(r);
  bool position = false;
  if (c.output_total != 0) {
    for (uint32_t attribute = 0; attribute < std::min(c.output_total, 7u); ++attribute)
      for (int component = 0; component < 4; ++component) {
        const uint32_t semantic = c.output_map[attribute] >> (component * 8) & 0x1F;
        if (semantic >= 24) continue;
        Slot& slot = s.slots[semantic];
        slot.reg = attribute < enabled.size() ? enabled[attribute] : -2;
        slot.component = component;
        position |= semantic <= 3;
      }
  }
  if (!position) {
    // the layout used when the registers describe nothing
    s = Semantics{};
    for (int i = 0; i < 4; ++i) s.slots[i] = {0, i}, s.slots[8 + i] = {2, i};
    s.slots[12] = {3, 0};
    s.slots[13] = {3, 1};
  }
  return s;
}

std::string pick(const Slot& slot, const char* fallback = "0.0") {
  if (slot.reg < 0) return fallback;
  return "o[" + std::to_string(slot.reg) + "]." + "xyzw"[slot.component];
}

// TEV
std::string tev_source(uint32_t source, int stage, bool lighting) {
  switch (source & 0xF) {
  case 0x0: return "color";
  case 0x1: return lighting ? "lit_primary" : "color";
  case 0x2: return lighting ? "lit_secondary" : "vec4f(0.0, 0.0, 0.0, 1.0)";
  case 0x3: return "t0";
  case 0x4: return "t1";
  case 0x5: return "t2";
  case 0x6: return "t3";
  case 0xD: return "tev_buffer";
  case 0xE: return "dr.tev_constant[" + std::to_string(stage) + "]";
  default: return "prev";
  }
}

std::string color_operand(const std::string& s, uint32_t operand) {
  switch (operand) {
  case 0x1: return "(vec3f(1.0) - " + s + ".rgb)";
  case 0x2: return "vec3f(" + s + ".a)";
  case 0x3: return "vec3f(1.0 - " + s + ".a)";
  case 0x4: return "vec3f(" + s + ".r)";
  case 0x5: return "vec3f(1.0 - " + s + ".r)";
  case 0x8: return "vec3f(" + s + ".g)";
  case 0x9: return "vec3f(1.0 - " + s + ".g)";
  case 0xC: return "vec3f(" + s + ".b)";
  case 0xD: return "vec3f(1.0 - " + s + ".b)";
  default: return s + ".rgb";
  }
}

std::string alpha_operand(const std::string& s, uint32_t operand) {
  switch (operand & 7) {
  case 0: return s + ".a";
  case 1: return "(1.0 - " + s + ".a)";
  case 2: return s + ".r";
  case 3: return "(1.0 - " + s + ".r)";
  case 4: return s + ".g";
  case 5: return "(1.0 - " + s + ".g)";
  case 6: return s + ".b";
  default: return "(1.0 - " + s + ".b)";
  }
}

template <typename T>
std::string combine(uint32_t op, const std::string& a, const std::string& b, const std::string& c, const T& one,
                    const std::string& zero) {
  switch (op & 0xF) {
  case 0: return a;
  case 1: return a + " * " + b;
  case 2: return "min(" + a + " + " + b + ", " + one + ")";
  case 3: return "clamp(" + a + " + " + b + " - " + one + " * 0.5, " + zero + ", " + one + ")";
  case 4: return a + " * " + c + " + " + b + " * (" + one + " - " + c + ")";
  case 5: return "max(" + a + " - " + b + ", " + zero + ")";
  case 8: return "min(" + a + " * " + b + " + " + c + ", " + one + ")";
  case 9: return "min(" + a + " + " + b + ", " + one + ") * " + c;
  default: return a; // dot3 is handled by the caller
  }
}

float scale_factor(uint32_t raw) { return (raw & 3) == 1 ? 2.0f : (raw & 3) == 2 ? 4.0f : 1.0f; }

bool passthrough(const Config& c, int stage) {
  const uint32_t source = c.tev[stage][0], operand = c.tev[stage][1], combiner = c.tev[stage][2];
  auto previous = [](uint32_t s) { return (s & 0xF) >= 0x7 && (s & 0xF) != 0xD && (s & 0xF) != 0xE; };
  return (combiner & 0xF) == 0 && (combiner >> 16 & 0xF) == 0 && previous(source) && previous(source >> 16) &&
         (operand & 0xF) == 0 && (operand >> 12 & 7) == 0 && scale_factor(c.tev_scale[stage]) == 1.0f &&
         scale_factor(c.tev_scale[stage] >> 16) == 1.0f;
}

bool reads_lighting(const Config& c) {
  for (int stage = 0; stage < 6; ++stage) {
    if (passthrough(c, stage)) continue;
    const uint32_t combiner = c.tev[stage][2], source = c.tev[stage][0];
    const uint32_t color_op = combiner & 0xF, alpha_op = combiner >> 16 & 0xF;
    auto inputs = [](uint32_t op) { return op == 0 ? 1 : (op == 4 || op == 8 || op == 9) ? 3 : 2; };
    auto lit = [](uint32_t s) { return (s & 0xF) == 1 || (s & 0xF) == 2; };
    for (int i = 0; i < inputs(color_op); ++i)
      if (lit(source >> (i * 4))) return true;
    if (color_op != 7)
      for (int i = 0; i < inputs(alpha_op); ++i)
        if (lit(source >> (16 + i * 4))) return true;
  }
  return false;
}

// lighting tables and which configurations read them
constexpr int kDistribution0 = 0, kDistribution1 = 1, kFresnel = 3, kReflectBlue = 4, kReflectGreen = 5,
              kReflectRed = 6, kSpotlight = 8, kDistance = 16;

bool supported(uint32_t config, int table) {
  switch (table) {
  case kDistribution0: return config != 1;
  case kDistribution1: return !(config == 0 || config == 1 || config == 5);
  case kSpotlight: return !(config == 2 || config == 3);
  case kFresnel: return !(config == 0 || config == 2 || config == 4);
  case kReflectRed: return config != 3;
  default: return config == 4 || config == 5 || config == 8;
  }
}

struct Lookup {
  bool on = false;
  uint32_t input = 0;
  bool absolute = true;
  float scale = 1.0f;
};

Lookup lookup(const Config& c, int field) {
  Lookup l;
  l.on = true;
  l.input = c.lut_input >> (field * 4) & 7;
  l.absolute = (c.lut_absolute >> (field * 4 + 1) & 1) == 0;
  static const float scales[8] = {1.0f, 2.0f, 4.0f, 8.0f, 0.0f, 0.0f, 0.25f, 0.5f};
  l.scale = scales[c.lut_scale >> (field * 4) & 7];
  return l;
}

std::string lighting_function(const Config& c) {
  const uint32_t config = c.light_config0 >> 4 & 0xF;
  auto table = [&](int t, int disabled, int field) {
    Lookup l;
    if ((c.light_config1 >> disabled & 1) == 0 && supported(config, t)) l = lookup(c, field);
    return l;
  };
  const Lookup d0 = table(kDistribution0, 16, 0), d1 = table(kDistribution1, 17, 1), fresnel = table(kFresnel, 19, 3);
  const Lookup reflect[3] = {table(kReflectRed, 20, 6), table(kReflectGreen, 21, 5), table(kReflectBlue, 22, 4)};
  const Lookup spot = lookup(c, 2);
  bool any_spot = false, geometric = false;
  for (uint32_t slot = 0; slot < c.light_count; ++slot) {
    any_spot |= (c.light_flags[slot] >> 5 & 1) != 0;
    geometric |= (c.light_flags[slot] & 0xC) != 0;
  }
  bool half_input = false, view_input = false;
  auto consider = [&](const Lookup& l) {
    if (!l.on) return;
    half_input |= l.input == 0 || l.input == 1 || (l.input == 5 && config == 8);
    view_input |= l.input == 1 || l.input == 2;
  };
  consider(d0), consider(d1), consider(fresnel);
  for (const auto& l : reflect) consider(l);
  if (any_spot) consider(spot);
  const bool needs_half = geometric || half_input, needs_view = needs_half || view_input;
  const uint32_t bump = c.light_config0 >> 28 & 3, bump_unit = c.light_config0 >> 22 & 3;
  const bool rebuild = (c.light_config0 >> 30 & 1) == 0;
  const bool shadow = c.light_config0 & 1;
  const uint32_t shadow_unit = c.light_config0 >> 24 & 3;
  const bool shadow_invert = c.light_config0 >> 18 & 1, shadow_primary = c.light_config0 >> 16 & 1,
             shadow_secondary = c.light_config0 >> 17 & 1, shadow_alpha = c.light_config0 >> 19 & 1;
  const bool fresnel_primary = c.light_config0 >> 2 & 1, fresnel_secondary = c.light_config0 >> 3 & 1,
             clamp_highlights = c.light_config0 >> 27 & 1;

  std::ostringstream w;
  w << "fn lut(table: u32, entry: u32, delta: f32) -> f32 { let e = lut_data[dr.lut.x + table * 256u + entry]; "
       "return e.x + e.y * delta; }\n";
  w << "fn rotate(q: vec4f, v: vec3f) -> vec3f { let inner = cross(q.xyz, v); return v + 2.0 * cross(q.xyz, inner + v * q.w); }\n";
  w << "fn shade(quaternion_in: vec4f, view: vec3f, t0: vec4f, t1: vec4f, t2: vec4f, t3: vec4f) -> array<vec4f, 2> {\n";
  w << "  let ql = length(quaternion_in);\n";
  w << "  let q = select(vec4f(0.0, 0.0, 0.0, 1.0), quaternion_in / ql, ql > 0.0);\n";
  w << "  let textures = array<vec4f, 4>(t0, t1, t2, t3);\n";
  if (shadow)
    w << "  let shadow = " << (shadow_invert ? "vec4f(1.0) - " : "") << "textures[" << shadow_unit << "];\n";
  else
    w << "  let shadow = vec4f(1.0);\n";
  if (bump == 1) {
    w << "  var surface_normal = textures[" << bump_unit << "].xyz * 2.0 - 1.0;\n";
    if (rebuild) w << "  surface_normal.z = sqrt(max(1.0 - dot(surface_normal.xy, surface_normal.xy), 0.0));\n";
    w << "  let surface_tangent = vec3f(1.0, 0.0, 0.0);\n";
  } else if (bump == 2) {
    w << "  let surface_normal = vec3f(0.0, 0.0, 1.0);\n  let surface_tangent = textures[" << bump_unit
      << "].xyz * 2.0 - 1.0;\n";
  } else {
    w << "  let surface_normal = vec3f(0.0, 0.0, 1.0);\n  let surface_tangent = vec3f(1.0, 0.0, 0.0);\n";
  }
  w << "  let normal = rotate(q, surface_normal);\n";
  w << "  let tangent = " << (config == 8 ? "rotate(q, surface_tangent)" : "surface_tangent") << ";\n";
  w << "  let norm_view = " << (needs_view ? "normalize(view)" : "vec3f(0.0)") << ";\n";
  w << "  var diffuse_sum = vec4f(0.0, 0.0, 0.0, 1.0);\n  var specular_sum = vec4f(0.0, 0.0, 0.0, 1.0);\n";
  for (uint32_t slot = 0; slot < c.light_count; ++slot) {
    const uint32_t flags = c.light_flags[slot];
    const bool directional = flags & 1, two_sided = flags >> 1 & 1, geometric0 = flags >> 2 & 1,
               geometric1 = flags >> 3 & 1, distance = flags >> 4 & 1, spotlight = flags >> 5 & 1,
               shadowed = flags >> 6 & 1;
    const uint32_t number = c.light_slots >> (slot * 4) & 7;
    const std::string L = "dr.lights[" + std::to_string(slot) + "]";
    w << "  {\n";
    w << "    let light_vector = " << (directional ? L + "[5].xyz" : "normalize(" + L + "[4].xyz + view)") << ";\n";
    if (needs_half) w << "    let half_v = norm_view + light_vector;\n    let half_unit = normalize(half_v);\n";
    else w << "    let half_v = vec3f(0.0);\n    let half_unit = vec3f(0.0);\n";
    if (distance) {
      w << "    let offset = -view - " << L << "[4].xyz;\n";
      w << "    let place = clamp(" << L << "[7].y * length(offset) + " << L << "[7].x, 0.0, 1.0);\n";
      w << "    let dentry = clamp(floor(place * 256.0), 0.0, 255.0);\n";
      w << "    let attenuation = lut(" << u(kDistance + number) << ", u32(dentry), place * 256.0 - dentry);\n";
    } else {
      w << "    let attenuation = 1.0;\n";
    }
    auto value = [&](const Lookup& l, const std::string& table) {
      std::string input;
      switch (l.input) {
      case 0: input = "dot(normal, half_unit)"; break;
      case 1: input = "dot(norm_view, half_unit)"; break;
      case 2: input = "dot(normal, norm_view)"; break;
      case 3: input = "dot(light_vector, normal)"; break;
      case 4: input = "dot(light_vector, " + L + "[6].xyz)"; break;
      case 5: input = config == 8 ? "dot(half_unit - normal * dot(normal, half_unit), tangent)" : "0.0"; break;
      default: input = "0.0";
      }
      std::string e;
      if (l.absolute) {
        e = "lut_abs(" + input + ", " + (two_sided ? "true" : "false") + ", " + table + ")";
      } else {
        e = "lut_signed(" + input + ", " + table + ")";
      }
      return f(l.scale) + " * " + e;
    };
    w << "    let spot = " << (spotlight ? value(spot, u(kSpotlight + number)) : "1.0") << ";\n";
    w << "    let d0 = " << (d0.on ? value(d0, u(kDistribution0)) : "1.0") << ";\n";
    w << "    let red = " << (reflect[0].on ? value(reflect[0], u(kReflectRed)) : "1.0") << ";\n";
    w << "    let refl = vec3f(red, " << (reflect[1].on ? value(reflect[1], u(kReflectGreen)) : "red") << ", "
      << (reflect[2].on ? value(reflect[2], u(kReflectBlue)) : "red") << ");\n";
    w << "    let d1 = " << (d1.on ? value(d1, u(kDistribution1)) : "1.0") << ";\n";
    w << "    var specular0 = " << L << "[0].xyz * d0;\n";
    w << "    var specular1 = d1 * refl * " << L << "[1].xyz;\n";
    if (slot == c.light_count - 1 && fresnel.on) {
      w << "    let fresnel = " << value(fresnel, u(kFresnel)) << ";\n";
      if (fresnel_primary) w << "    diffuse_sum.w = fresnel;\n";
      if (fresnel_secondary) w << "    specular_sum.w = fresnel;\n";
    }
    w << "    let facing = " << (two_sided ? "abs(dot(light_vector, normal))" : "max(dot(light_vector, normal), 0.0)")
      << ";\n";
    w << "    let highlights = " << (clamp_highlights ? "select(1.0, 0.0, facing == 0.0)" : "1.0") << ";\n";
    if (geometric0 || geometric1) {
      w << "    let hl = dot(half_v, half_v);\n";
      w << "    let gfactor = select(min(facing / hl, 1.0), 0.0, hl == 0.0);\n";
      if (geometric0) w << "    specular0 = specular0 * gfactor;\n";
      if (geometric1) w << "    specular1 = specular1 * gfactor;\n";
    }
    const bool sp = shadow && shadow_primary && shadowed, ss = shadow && shadow_secondary && shadowed;
    w << "    let diffuse = (" << L << "[2].xyz * facing + " << L << "[3].xyz) * attenuation * spot;\n";
    w << "    let specular = (specular0 + specular1) * highlights * attenuation * spot;\n";
    w << "    diffuse_sum = vec4f(diffuse_sum.xyz + diffuse" << (sp ? " * shadow.xyz" : "") << ", diffuse_sum.w);\n";
    w << "    specular_sum = vec4f(specular_sum.xyz + specular" << (ss ? " * shadow.xyz" : "")
      << ", specular_sum.w);\n";
    w << "  }\n";
  }
  if (shadow && shadow_alpha) {
    if (fresnel_primary) w << "  diffuse_sum.w = diffuse_sum.w * shadow.w;\n";
    if (fresnel_secondary) w << "  specular_sum.w = specular_sum.w * shadow.w;\n";
  }
  w << "  diffuse_sum = vec4f(diffuse_sum.xyz + dr.global_ambient.xyz, diffuse_sum.w);\n";
  // the unit hands the combiners 8-bit colors
  w << "  return array<vec4f, 2>(floor(clamp(diffuse_sum, vec4f(0.0), vec4f(1.0)) * 255.0) / 255.0, "
       "floor(clamp(specular_sum, vec4f(0.0), vec4f(1.0)) * 255.0) / 255.0);\n}\n";
  w << "fn lut_abs(x: f32, two_sided: bool, table: u32) -> f32 {\n"
       "  let r = select(max(x, 0.0), abs(x), two_sided);\n"
       "  let e = clamp(floor(r * 256.0), 0.0, 255.0);\n  return lut(table, u32(e), r * 256.0 - e);\n}\n";
  // a signed index wraps into the upper half of the table
  w << "fn lut_signed(x: f32, table: u32) -> f32 {\n"
       "  let e = clamp(floor(x * 128.0), -128.0, 127.0);\n"
       "  return lut(table, u32(i32(e) & 0xFF), x * 128.0 - e);\n}\n";
  return w.str();
}

// A texture unit's lookup: the sampler bound with it wraps and filters (see
// Renderer::sampler); the border color and the level of detail are the
// shader's. Rows are stored top down, so t runs the other way.
std::string texture_functions(int unit) {
  const std::string n = std::to_string(unit);
  std::ostringstream w;
  w << "fn sample" << n << "(uv: vec2f, rate: f32) -> vec4f {\n"
    << "  let size = dr.texture_size[" << n << "];\n"
    << "  let st = vec2f(uv.x, 1.0 - uv.y);\n"
    << "  let ws = (size.w >> 12u) & 7u;\n  let wt = (size.w >> 8u) & 7u;\n"
    << "  if ((border(ws) && (st.x < 0.0 || st.x >= 1.0)) || (border(wt) && (st.y < 0.0 || st.y >= 1.0))) { return "
       "dr.texture_border["
    << n << "]; }\n"
    << "  var level = 0.0;\n"
    << "  if (size.z > 1u) {\n"
    << "    let lod = dr.texture_lod[" << n << "];\n"
    << "    let l = clamp(rate + lod.x, lod.y, lod.z);\n"
    << "    level = select(0.0, l, l > 0.0);\n  }\n"
    << "  return textureSampleLevel(tex" << n << ", smp" << n << ", st, level);\n}\n";
  return w.str();
}

} // namespace

std::vector<uint32_t> Config::key() const {
  std::vector<uint32_t> k = {static_cast<uint32_t>(program), static_cast<uint32_t>(program >> 32), entry, output_mask,
                             output_total};
  k.insert(k.end(), output_map, output_map + 7);
  k.push_back(loader_count);
  k.insert(k.end(), loader_fields, loader_fields + 12);
  for (const Field& field : fields) k.insert(k.end(), {field.offset, field.type, field.count, field.input});
  k.insert(k.end(), {topology, indexed, upside_down, textures, texture0_projected, texture2_coord1});
  for (int s = 0; s < 6; ++s) k.insert(k.end(), {tev[s][0], tev[s][1], tev[s][2], tev_scale[s]});
  k.insert(k.end(), {buffer_update, alpha_test, fog, fog_flip, lighting, light_count, light_slots, light_config0,
                     light_config1});
  k.insert(k.end(), light_flags, light_flags + 8);
  k.insert(k.end(), {lut_absolute, lut_input, lut_scale, logic_mode, blend, color_mask, depth_test, depth_write,
                     stencil, stencil_test, stencil_op, cull});
  return k;
}

Config configure(const pica::State& state, const vertex::Layout& layout, bool indexed) {
  const auto& r = state.regs;
  Config c;
  c.program = shadergen::program_key(state.vs);
  c.entry = state.vs.entry_point;
  c.output_mask = r[pica::kRegVsBlock + 0x0D] & 0xFFFF;
  c.output_total = r[0x04F] & 7;
  for (int i = 0; i < 7; ++i) c.output_map[i] = r[0x050 + i];
  c.loader_count = static_cast<uint32_t>(layout.loaders.size());
  for (size_t l = 0; l < layout.loaders.size(); ++l) {
    c.loader_fields[l] = static_cast<uint32_t>(layout.loaders[l].fields.size());
    for (const auto& field : layout.loaders[l].fields) c.fields.push_back({field.offset, field.type, field.count, field.input});
  }
  c.topology = r[0x25E] >> 8 & 3;
  c.indexed = indexed;
  c.upside_down = (r[0x11E] >> 24 & 1) == 0;

  c.textures = r[0x080] & 7;
  c.texture0_projected = (r[0x083] >> 28 & 7) == 3;
  c.texture2_coord1 = r[0x080] >> 13 & 1;
  static const uint32_t stages[6] = {0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8};
  for (int s = 0; s < 6; ++s) {
    c.tev[s][0] = r[stages[s]] & 0x0FFF0FFF;
    c.tev[s][1] = r[stages[s] + 1] & 0x00777FFF;
    c.tev[s][2] = r[stages[s] + 2] & 0x000F000F;
    c.tev_scale[s] = r[stages[s] + 4] & 0x00030003;
  }
  c.buffer_update = r[0x0E0] & 0xFF00;
  c.alpha_test = r[0x104] & 1 ? (r[0x104] >> 4 & 7) + 1 : 0;
  c.fog = (r[0x0E0] & 7) == 5;
  c.fog_flip = c.fog && (r[0x0E0] >> 16 & 1);
  c.lighting = (r[0x08F] & 1) && !(r[0x1C6] & 1) && reads_lighting(c);
  if (c.lighting) {
    c.light_count = (r[0x1C2] & 7) + 1;
    c.light_slots = r[0x1D9] & 0x77777777;
    c.light_config0 = r[0x1C3];
    c.light_config1 = r[0x1C4];
    const uint32_t config = c.light_config0 >> 4 & 0xF;
    for (uint32_t slot = 0; slot < c.light_count; ++slot) {
      const uint32_t number = c.light_slots >> (slot * 4) & 7;
      const uint32_t block9 = r[0x140 + number * 0x10 + 9];
      uint32_t flags = block9 & 0xF;
      if ((c.light_config1 >> (24 + number) & 1) == 0) flags |= 1 << 4;
      if ((c.light_config1 >> (8 + number) & 1) == 0 && supported(config, kSpotlight)) flags |= 1 << 5;
      if ((c.light_config1 >> number & 1) == 0) flags |= 1 << 6;
      c.light_flags[slot] = flags;
    }
    c.lut_absolute = r[0x1D0];
    c.lut_input = r[0x1D1];
    c.lut_scale = r[0x1D2];
  }
  const bool blending = r[0x100] >> 8 & 1;
  c.logic_mode = blending ? 0 : (r[0x102] & 0xF) + 1;
  if (blending) c.blend = r[0x101];
  c.color_mask = r[0x113] != 0 ? (r[0x107] >> 8 & 0xF) : 0;
  const bool writable = r[0x115] != 0;
  c.depth_test = r[0x107] & 1 ? (r[0x107] >> 4 & 7) + 1 : 0;
  c.depth_write = writable && (r[0x107] >> 12 & 1);
  if ((r[0x116] & 3) == 3) {
    const bool enabled = r[0x105] & 1;
    const bool changes = writable && (r[0x105] >> 8 & 0xFF) != 0 && (r[0x106] & 0x777) != 0;
    c.stencil = enabled || changes;
    if (c.stencil) {
      c.stencil_test = (enabled ? (r[0x105] >> 4 & 7) : 1) | (writable ? r[0x105] & 0xFF00 : 0) |
                       (r[0x105] & 0xFF000000);
      c.stencil_op = r[0x106] & 0x777;
    }
  }
  c.cull = r[0x040] & 3;
  return c;
}

namespace {

const char* kDrawStruct = R"(struct Draw {
  loaders: array<vec4u, 12>,
  index: vec4u,
  defaults: array<vec4f, 16>,
  clip: vec4f,
  depth: vec4f,
  tev_constant: array<vec4f, 6>,
  buffer_color: vec4f,
  fog_color: vec4f,
  alpha_reference: vec4u,
  texture_size: array<vec4u, 3>,
  texture_border: array<vec4f, 3>,
  texture_lod: array<vec4f, 3>,
  lut: vec4u,
  global_ambient: vec4f,
  lights: array<array<vec4f, 9>, 8>,
}
@group(0) @binding(1) var<uniform> dr: Draw;
)";

const char* kVOut = R"(struct VOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
  @location(1) uv01: vec4f,
  @location(2) uv2: vec2f,
  @location(3) quaternion: vec4f,
  @location(4) view: vec3f,
  @location(5) @interpolate(linear) z: f32,
  @location(6) uv0w: f32,
  @location(7) w: f32,
}
)";

} // namespace

std::string vertex_wgsl(const Config& c, const pica::ShaderUnit& vs) {
  std::ostringstream w;
  w << shadergen::kVertexUniformsWgsl;
  w << kDrawStruct;
  w << "@group(0) @binding(0) var<uniform> vsu: PicaVsUniforms;\n"
       "@group(0) @binding(2) var<storage, read> arena: array<u32>;\n";
  w << shadergen::vertex_function(vs, "vsu");
  w << kVOut;
  w << "fn rb(a: u32) -> u32 { return (arena[a >> 2u] >> ((a & 3u) * 8u)) & 0xFFu; }\n"
       "fn r16(a: u32) -> u32 { return rb(a) | (rb(a + 1u) << 8u); }\n"
       "fn r32(a: u32) -> u32 { return r16(a) | (r16(a + 2u) << 16u); }\n";

  // vertex stage
  w << "@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {\n";
  w << "  let t = vi / 3u;\n  let k = vi % 3u;\n";
  switch (c.topology) {
  case 1: w << "  let p = select(select(select(t + 2u, t, k == 1u), t + 1u, k == 0u), t + k, (t & 1u) == 0u);\n"; break;
  case 2: w << "  let p = select(t + k, 0u, k == 0u);\n"; break;
  default: w << "  let p = vi;\n"; break;
  }
  if (c.indexed) w << "  let id = select(rb(dr.index.x + p), r16(dr.index.x + 2u * p), dr.index.y != 0u);\n";
  else w << "  let id = dr.index.z + p;\n";
  w << "  var v: array<vec4f, 16>;\n";
  for (int i = 0; i < 16; ++i) w << "  v[" << i << "] = dr.defaults[" << i << "];\n";
  size_t field_index = 0;
  for (uint32_t l = 0; l < c.loader_count; ++l) {
    w << "  {\n    let base = dr.loaders[" << l << "].x + id * dr.loaders[" << l << "].y;\n";
    for (uint32_t i = 0; i < c.loader_fields[l]; ++i, ++field_index) {
      const Field& fd = c.fields[field_index];
      std::string comps[4] = {"0.0", "0.0", "0.0", "1.0"};
      for (uint32_t k = 0; k < fd.count; ++k) {
        const uint32_t size = fd.type <= 1 ? 1 : fd.type == 2 ? 2 : 4;
        const std::string at = "base + " + u(fd.offset + k * size);
        switch (fd.type) {
        case 0: comps[k] = "f32(i32(rb(" + at + ") << 24u) >> 24u)"; break;
        case 1: comps[k] = "f32(rb(" + at + "))"; break;
        case 2: comps[k] = "f32(i32(r16(" + at + ") << 16u) >> 16u)"; break;
        default: comps[k] = "bitcast<f32>(r32(" + at + "))"; break;
        }
      }
      w << "    v[" << fd.input << "] = vec4f(" << comps[0] << ", " << comps[1] << ", " << comps[2] << ", "
        << comps[3] << ");\n";
    }
    w << "  }\n";
  }
  w << "  let o = pica_vs(v);\n";
  const Semantics s = semantics(c);
  w << "  let pos = vec4f(" << pick(s.slots[0]) << ", " << pick(s.slots[1]) << ", " << pick(s.slots[2]) << ", "
    << pick(s.slots[3]) << ");\n";
  w << "  var vout: VOut;\n";
  const std::string y_sign = c.upside_down ? "-" : "";
  w << "  vout.position = vec4f(dr.clip.x * pos.x + dr.clip.y * pos.w, " << y_sign
    << "(dr.clip.z * pos.y + dr.clip.w * pos.w), pos.z + pos.w, pos.w);\n";
  w << "  vout.z = pos.z / pos.w;\n  vout.w = pos.w;\n";
  w << "  vout.color = vec4f(" << pick(s.slots[8]) << ", " << pick(s.slots[9]) << ", " << pick(s.slots[10]) << ", "
    << pick(s.slots[11]) << ");\n";
  w << "  vout.uv01 = vec4f(" << pick(s.slots[12]) << ", " << pick(s.slots[13]) << ", " << pick(s.slots[14]) << ", "
    << pick(s.slots[15]) << ");\n";
  w << "  vout.uv2 = vec2f(" << pick(s.slots[22]) << ", " << pick(s.slots[23]) << ");\n";
  w << "  vout.uv0w = " << pick(s.slots[16]) << ";\n";
  w << "  vout.quaternion = vec4f(" << pick(s.slots[4]) << ", " << pick(s.slots[5]) << ", " << pick(s.slots[6])
    << ", " << pick(s.slots[7]) << ");\n";
  w << "  vout.view = vec3f(" << pick(s.slots[18]) << ", " << pick(s.slots[19]) << ", " << pick(s.slots[20]) << ");\n";
  w << "  return vout;\n}\n";
  return w.str();
}

std::string fragment_wgsl(const Config& c) {
  std::ostringstream w;
  w << kDrawStruct;
  w << "@group(0) @binding(3) var<storage, read> lut_data: array<vec2f>;\n"
       "@group(0) @binding(4) var tex0: texture_2d<f32>;\n"
       "@group(0) @binding(5) var tex1: texture_2d<f32>;\n"
       "@group(0) @binding(6) var tex2: texture_2d<f32>;\n"
       "@group(0) @binding(7) var smp0: sampler;\n"
       "@group(0) @binding(8) var smp1: sampler;\n"
       "@group(0) @binding(9) var smp2: sampler;\n";
  w << kVOut;
  // fragment stage
  w << "fn border(mode: u32) -> bool { return mode == 1u || mode == 5u; }\n"
       "fn compare(function: u32, a: f32, b: f32) -> bool {\n"
       "  switch function {\n    case 0u: { return false; }\n    case 1u: { return true; }\n"
       "    case 2u: { return a == b; }\n    case 3u: { return a != b; }\n    case 4u: { return a < b; }\n"
       "    case 5u: { return a <= b; }\n    case 6u: { return a > b; }\n    default: { return a >= b; }\n  }\n}\n";
  for (int unit = 0; unit < 3; ++unit)
    if (c.textures >> unit & 1) w << texture_functions(unit);
  if (c.lighting) w << lighting_function(c);
  w << "struct FOut {\n  @location(0) color: vec4f,\n  @builtin(frag_depth) depth: f32,\n}\n";
  w << "@fragment fn fs(inp: VOut) -> FOut {\n";
  w << "  let color = clamp(inp.color, vec4f(0.0), vec4f(1.0));\n";
  w << "  var t0 = vec4f(0.0, 0.0, 0.0, 1.0);\n  var t1 = t0;\n  var t2 = t0;\n  let t3 = t0;\n";
  // the coordinates and their rates of change first, while control flow is uniform
  for (int unit = 0; unit < 3; ++unit) {
    if (!(c.textures >> unit & 1)) continue;
    const std::string n = std::to_string(unit);
    std::string uv = unit == 0 ? "inp.uv01.xy" : unit == 1 ? "inp.uv01.zw" : (c.texture2_coord1 ? "inp.uv01.zw" : "inp.uv2");
    if (unit == 0 && c.texture0_projected) uv = "(inp.uv01.xy / inp.uv0w)";
    w << "  let uv" << n << " = " << uv << ";\n";
    w << "  let scaled" << n << " = uv" << n << " * vec2f(f32(dr.texture_size[" << n << "].x), f32(dr.texture_size["
      << n << "].y));\n";
    w << "  let dx" << n << " = dpdx(scaled" << n << ");\n  let dy" << n << " = dpdy(scaled" << n << ");\n";
    w << "  let rate" << n << " = 0.5 * log2(max(dot(dx" << n << ", dx" << n << "), dot(dy" << n << ", dy" << n
      << ")));\n";
  }
  for (int unit = 0; unit < 3; ++unit)
    if (c.textures >> unit & 1)
      w << "  t" << unit << " = sample" << unit << "(uv" << unit << ", rate" << unit << ");\n";
  if (c.lighting) {
    w << "  let lit = shade(inp.quaternion, inp.view, t0, t1, t2, t3);\n";
    w << "  let lit_primary = lit[0];\n  let lit_secondary = lit[1];\n";
  }
  // the texture environment: the buffer lags a stage behind, the first
  // stage reads zero, the second the configured buffer color
  w << "  var prev = color;\n  var tev_buffer = vec4f(0.0);\n  var next_buffer = dr.buffer_color;\n";
  for (int stage = 0; stage < 6; ++stage) {
    if (!passthrough(c, stage)) {
      const uint32_t source = c.tev[stage][0], operand = c.tev[stage][1], combiner = c.tev[stage][2];
      const uint32_t color_op = combiner & 0xF, alpha_op = combiner >> 16 & 0xF;
      std::string rgb[3], alpha[3];
      for (int i = 0; i < 3; ++i) {
        rgb[i] = color_operand(tev_source(source >> (i * 4), stage, c.lighting), operand >> (i * 4) & 0xF);
        alpha[i] = alpha_operand(tev_source(source >> (16 + i * 4), stage, c.lighting), operand >> (12 + i * 4) & 7);
      }
      w << "  {\n    let r0 = " << rgb[0] << ";\n    let r1 = " << rgb[1] << ";\n    let r2 = " << rgb[2] << ";\n";
      w << "    let a0 = " << alpha[0] << ";\n    let a1 = " << alpha[1] << ";\n    let a2 = " << alpha[2] << ";\n";
      if (color_op == 6 || color_op == 7)
        w << "    let rgb = vec3f(clamp(dot(r0 * 2.0 - 1.0, r1 * 2.0 - 1.0), 0.0, 1.0));\n";
      else
        w << "    let rgb = " << combine(color_op, "r0", "r1", "r2", std::string("vec3f(1.0)"), "vec3f(0.0)") << ";\n";
      if (color_op == 7) w << "    let alpha = rgb.r;\n";
      else if (alpha_op == 6 || alpha_op == 7) w << "    let alpha = a0;\n";
      else w << "    let alpha = " << combine(alpha_op, "a0", "a1", "a2", std::string("1.0"), "0.0") << ";\n";
      w << "    prev = clamp(vec4f(rgb * " << f(scale_factor(c.tev_scale[stage])) << ", alpha * "
        << f(scale_factor(c.tev_scale[stage] >> 16)) << "), vec4f(0.0), vec4f(1.0));\n  }\n";
    }
    w << "  tev_buffer = next_buffer;\n";
    if (stage < 4) {
      if (c.buffer_update >> (8 + stage) & 1) w << "  next_buffer = vec4f(prev.rgb, next_buffer.a);\n";
      if (c.buffer_update >> (12 + stage) & 1) w << "  next_buffer.a = prev.a;\n";
    }
  }
  // what the combiners hand on is 8-bit
  w << "  var px = floor(prev * 255.0);\n";
  if (c.alpha_test) w << "  if (!compare(" << u(c.alpha_test - 1) << ", px.a, f32(dr.alpha_reference.x))) { discard; }\n";
  w << "  var d = inp.z * dr.depth.x + dr.depth.y;\n  if (dr.depth.z != 0.0) { d = d * inp.w; }\n";
  w << "  d = clamp(d, 0.0, 1.0);\n";
  if (c.fog) {
    w << "  let fog_index = " << (c.fog_flip ? "(1.0 - d)" : "d") << " * 128.0;\n";
    w << "  let fog_entry = clamp(floor(fog_index), 0.0, 127.0);\n";
    w << "  let fog_e = lut_data[dr.lut.y + u32(fog_entry)];\n";
    w << "  let fog = clamp(fog_e.x + fog_e.y * (fog_index - fog_entry), 0.0, 1.0);\n";
    w << "  px = vec4f(floor(fog * px.rgb + (1.0 - fog) * dr.fog_color.rgb), px.a);\n";
  }
  switch (c.logic_mode) {
  case 1: w << "  px = vec4f(0.0);\n"; break;   // clear
  case 5: w << "  px = vec4f(255.0);\n"; break; // set
  default: break;
  }
  w << "  var fout: FOut;\n  fout.color = px / 255.0;\n  fout.depth = floor(d * 16777215.0) / 16777215.0;\n";
  w << "  return fout;\n}\n";
  return w.str();
}

} // namespace picanative::draw
