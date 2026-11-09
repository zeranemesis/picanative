# picanative — conception

Décidé le 9 novembre 2026 : le port de Super Mario 3D Land a **son propre moteur de rendu**, écrit
pour lui, le plus rapide et le plus modulaire possible. `zakuro-gpu` n'en fait
pas partie. Il sert seulement de **référence de comparaison** pendant le
développement, comme les fonctions d'origine servent de référence au banc de
décompilation.

## Ce que le moteur reçoit

Le code de rendu du jeu (NintendoWare, `nn::gx`) parle au GPU de la 3DS par le
service GSP. Le moteur remplace ce service au niveau de l'API du SDK, comme la
couche plateforme le fait déjà pour les fichiers et la manette.

| Entrée | Contenu |
|---|---|
| `ProcessCommandList(adresse, taille)` | une liste d'écritures de registres PICA : l'état du GPU et les dessins |
| `MemoryFill` | effacement des tampons couleur et profondeur |
| `DisplayTransfer` | copie d'un rendu vers une image d'écran, avec conversion de format et réduction |
| `TextureCopy` | copie de mémoire graphique, souvent un rendu réutilisé comme texture |
| `RequestDma`, `FlushCacheRegions` | mouvements de données entre la mémoire du jeu et la mémoire vidéo |
| échange d'images | quelle image afficher sur chaque écran, et quand |
| sortie : interruptions | fin de dessin, fin de copie, VBlank : le jeu les attend pour continuer |

## Architecture en modules

Chaque module est une bibliothèque C++20 avec une interface faite de
**données**, testable seul, sans GPU pour tous ceux du haut.

```
gx/          entrées GSP ci-dessus -> appels au moteur ; interruptions vers le jeu
pica/        registres PICA, analyse des listes de commandes, état décodé
             (aucun appel GPU : pur, déterministe)
vertex/      chargeurs d'attributs, tampons d'index, primitives
shadergen/   état décodé -> shaders :
               - shaders de sommets PICA (bytecode) -> SPIR-V
               - combineurs TEV, éclairage par fragment, brouillard,
                 tests alpha/stencil/profondeur -> shader de fragments
             (fonctions pures : même état, même shader)
texture/     décodage des 14 formats PICA (tuilés, ETC1/ETC1A4), cache par empreinte
surface/     cache des surfaces : rendu, profondeur, textures et écrans qui
             partagent la mémoire vidéo ; copie en mémoire du jeu seulement à la demande
gpu/         WebGPU (Dawn) : appareil, cache de pipelines, fil de rendu ;
             D3D12 ou Vulkan sur Windows, Vulkan sur Android
present/     les deux écrans : disposition, mise à l'échelle, 3D stéréo plus tard
trace/       enregistrement et relecture des entrées ci-dessus, image par image
```

Les modules ne connaissent que leurs voisins directs. Le choix de l'API graphique
(D3D12, Vulkan, Metal) est celui de Dawn : il ne touche ni `pica/` ni `shadergen/`, et tester
`shadergen/` ou `texture/` sans écran.

## Choix pour la vitesse

1. **Shaders préparés à l'avance.** Le jeu n'a qu'un nombre fini de shaders et
   d'états de fragments. Ceux rencontrés pendant les traces sont compilés une
   fois et gardés dans un cache sur disque (local, jamais publié). En jeu, on ne
   compile que ce qui manque, en arrière-plan, sans à-coups. Un shader de
   secours générique couvre l'attente.
2. **WGSL produit par `shadergen/`**, compilé par Dawn pour l'API de la
   machine, comme le fait Aurora. Le cache de pipelines est « transférable » :
   celui construit sur nos traces peut accompagner le port, et la Thor ne
   compile presque rien au lancement.
3. **État haché une fois par dessin** : pipeline, descripteurs et textures sont
   retrouvés par empreinte, jamais reconstruits si rien n'a changé.
4. **Pas de retour GPU → processeur** pendant l'image, sauf si le jeu lit
   vraiment la mémoire d'une surface. Le cache de surfaces le sait.
5. **Textures décodées sur le GPU** (calcul) pour les gros formats, mises en
   cache par empreinte de leur mémoire.
6. **Encodage des commandes GPU sur un fil de rendu séparé** du jeu, une image
   d'avance au plus, comme le `render_worker` d'Aurora.
7. Résolution interne réglable, 1× à 4× et plus : le jeu dessine en 400×240,
   le moteur peut dessiner plus fin.

## Vérification

Aucun module n'est déclaré fini sans comparaison automatique.

- **Traces.** `trace/` enregistre, pendant une vraie partie sur l'échafaudage,
  tout ce que le jeu envoie au GPU et la mémoire que les commandes lisent.
  Chaque trace peut ensuite être rejouée sans le jeu, en quelques secondes.
- **Images de référence.** Chaque trace est aussi rejouée dans `zakuro-gpu`.
  Notre image est comparée pixel par pixel, avec un seuil de tolérance noté pour
  chaque différence acceptée (arrondis d'interpolation, par exemple).
- **Tests de module.** Décodeurs de textures et traducteur de shaders comparés
  à une exécution de référence sur le processeur ; `pica/` comparé registre par
  registre.
- Jeu de traces de référence : écran titre, dialogues, sélection de fichier,
  cinématique, carte, 1-1. Puis on en ajoute à chaque niveau.

## Étapes

| Étape | Contenu | Jalon vérifiable |
|---|---|---|
| G0 | `trace/` : enregistreur dans l'échafaudage, relecteur, comparateur d'images | traces des 6 scènes de référence, relues à l'identique dans `zakuro-gpu` |
| G1 | `gx/`, `pica/`, `gpu/` de base, `present/` ; MemoryFill, DisplayTransfer, TextureCopy | les dialogues 2D de l'écran du bas identiques à la référence |
| G2 | `vertex/`, shaders de sommets → SPIR-V, TEV de base, textures courantes | logo et écran titre en 3D |
| G3 | TEV complet, tous les formats de texture, éclairage par fragment, brouillard, blending, stencil | cinématique et carte du monde |
| G4 | cache de surfaces complet (rendu vers texture), cas rares | le 1-1 entier identique à la référence |
| G5 | branchement dans l'hôte à la place de `zakuro-gpu`, puis la Thor | 1-1 jouable sur PC, puis sur la Thor |

## Modèle : Aurora

[Aurora](https://github.com/encounter/aurora) (MIT) est la couche de
compatibilité GameCube et Wii des ports décompilés : Metaforce, Dusklight,
Mario Party 4 et 6, Super Mario Sunshine. `picanative` suit son architecture
et sa pile, adaptées du GX au PICA200 :

| Aurora | picanative |
|---|---|
| SDL3 pour la fenêtre, les entrées et Android | pareil |
| Dawn / WebGPU : D3D12, Vulkan, Metal | pareil (`gpu/`) |
| `gx/command_processor` : lit les listes d'affichage GX | `pica/` : lit les listes de commandes PICA |
| `gx/shader` : état GX → WGSL | `shadergen/` : shaders PICA et TEV → WGSL |
| `gfx/pipeline_cache` : cache transférable, compilation asynchrone | pareil |
| `gfx/render_worker` : encodage sur un fil séparé | pareil |
| `gfx/texture_convert` : formats GameCube | `texture/` : formats PICA, ETC1 |
| `gfx/tex_copy_conv` : copies EFB vers texture | `surface/` : rendu vers texture, DisplayTransfer |

Le code générique d'Aurora (appareil Dawn, cache de pipelines, fil de rendu)
peut être repris en le créditant. Ce qui est propre au PICA200 est écrit ici.

## Ce qu'on reprend, ce qu'on écrit

Le comportement du PICA200 est documenté par la communauté (3dbrew) et par les
émulateurs. `zakuro-gpu` est sous licence MIT : on peut s'y référer pour un
détail matériel et reprendre une table en le créditant ; il sert surtout de
référence de comparaison. L'architecture suit Aurora.
