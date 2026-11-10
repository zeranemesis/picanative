# picanative

Moteur de rendu natif pour les ports PC et Android de jeux Nintendo 3DS
décompilés. Il reçoit ce que le code d'un jeu envoie au GPU PICA200 de la
console (listes de commandes, effacements, copies, échanges d'images) et le
dessine avec WebGPU (Dawn : D3D12 ou Vulkan sur Windows, Vulkan sur Android).
Son architecture suit celle d'[Aurora](https://github.com/encounter/aurora),
la couche GameCube des ports décompilés.

Premier client : le port natif de Super Mario 3D Land, sur PC et sur l'AYN
Thor.

**État : le moteur rend Super Mario 3D Land sur PC**, du démarrage au niveau 1-1,
comparé image par image à une implémentation de référence (voir les étapes dans
[DESIGN.md](DESIGN.md)), dans sa propre fenêtre SDL3. Reste : la Thor (Android).

## La fenêtre

Compilé avec SDL3, le moteur ouvre aussi la fenêtre du port (`pn_window_*`
dans `include/picanative/api.h`) : les deux écrans dessinés par son propre
device (directement depuis le GPU quand un transfert fait sur le GPU a produit
l'image, sans relecture), le clavier, les manettes (A à droite, comme sur la
3DS), l'écran tactile à la souris ou au doigt, et le son du DSP (32 728 Hz).
Une couche plateforme qui lit elle-même les entrées les retrouve avec
`pn_input_now`.

## Principes

- **Ports d'abord.** Le moteur sert un jeu décompilé et compilé nativement,
  pas un émulateur. Il remplace le service GPU de la console au niveau de l'API
  du SDK.
- **Modulaire.** Des modules C++20 qui ne communiquent que par des données :
  décodage de l'état PICA, génération de shaders, textures, surfaces, une petite
  couche WebGPU, comme dans Aurora.
- **Rapide.** Shaders WGSL générés, cache de pipelines transférable, fil de rendu séparé,
  aucun retour du GPU vers le processeur sans nécessité, résolution interne
  réglable.
- **Vérifié.** Chaque étape est comparée pixel par pixel, sur des traces
  enregistrées, à une implémentation de référence.

## Ce que ce dépôt ne contient pas

Ni ROM, ni code, ni données, ni traces tirées d'un jeu. Les traces servent aux
tests mais restent chez celui qui les enregistre, à partir de sa propre copie.

## Licence

MIT, voir [LICENSE](LICENSE).

## Compiler

```
cmake -S . -B build -DCMAKE_PREFIX_PATH=<paquet Dawn précompilé, celui d'Aurora> [-DSDL3_DIR=<SDL3 précompilé, celui d'Aurora>/cmake]
cmake --build build --config Release
build/Release/pn_optest <dossier de cas .pncase>
build/Release/pn_state <trace.pntrace> [états.txt] [dessins.txt]
build/Release/pn_shadertest <dossier de cas .pnvs>
build/Release/pn_texturetest <dossier de cas .pntx>
build/Release/pn_render [--cached] <trace.pntrace> [dossier d'images]
build/Release/pn_stats <trace.pntrace>
```
