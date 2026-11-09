# Format de trace `.pntrace`, version 1

Une trace contient tout ce qu'un jeu a envoyé au GPU PICA200 depuis son
démarrage, dans l'ordre, avec la mémoire que ces commandes lisent. Le moteur
peut la rejouer sans le jeu. Une trace est tirée d'un jeu : elle ne va jamais
dans ce dépôt.

## Pourquoi depuis le démarrage

Le GPU garde un état interne d'une image à l'autre : registres, programmes de
shaders, tables d'éclairage et de brouillard. Une trace qui commencerait en
cours de partie devrait reconstruire cet état. Enregistrer depuis le démarrage
évite ce problème : pour voir l'image 3 600, on rejoue les 3 599 précédentes.
Sans jeu à faire tourner, c'est rapide.

## Mémoire

Les adresses sont **physiques**, comme celles que voit le GPU :

| Zone | Adresse physique | Taille |
|---|---|---|
| VRAM | `0x18000000` | 6 Mio |
| FCRAM | `0x20000000` | 128 Mio (256 Mio sur New 3DS) |

Au départ, toute la mémoire vaut zéro. Avant chaque opération GPU,
l'enregistreur compare la mémoire à ce qu'elle était après l'opération
précédente, et écrit les pages de 4 Kio que **le processeur** a changées. Ce que
le GPU a lui-même écrit (rendus, copies) n'est pas enregistré : le moteur qui
rejoue doit le produire lui-même, c'est justement ce qu'on vérifie.

## Structure

Tout est en petit-boutiste. Le fichier commence par 16 octets :

| Octets | Contenu |
|---|---|
| 0-7 | `PNTRACE\0` |
| 8-11 | version, `1` |
| 12-15 | réservé, `0` |

Viennent ensuite des enregistrements, chacun avec un en-tête de 8 octets :
`type` (u32), puis `taille` (u32) de la charge qui suit.

| Type | Nom | Charge |
|---|---|---|
| 1 | `MEMORY` | `adresse` (u32, physique, multiple de 4096), puis les octets, par pages entières et contiguës |
| 2 | `REGISTER` | `décalage` (u32), `valeur` (u32) : écriture dans les registres externes du GPU (`0x1EF00000` + décalage), comme `gsp::WriteHWRegs` |
| 3 | `FRAMEBUFFER` | `écran` (0 haut, 1 bas), `actif`, `gauche`, `droite` (adresses physiques), `pas`, `format` : les six u32 de l'échange d'image |
| 4 | `COMMAND_LIST` | `adresse` (physique), `taille` en octets |
| 5 | `MEMORY_FILL` | `début`, `fin` (physiques), `valeur`, `largeur` (16, 24 ou 32 bits) |
| 6 | `DISPLAY_TRANSFER` | `entrée`, `sortie` (physiques), `dimensions d'entrée`, `dimensions de sortie`, `drapeaux` |
| 7 | `TEXTURE_COPY` | `entrée`, `sortie` (physiques), `taille`, `écart d'entrée`, `écart de sortie` |
| 8 | `FRAME` | `numéro` (u32) : fin de l'image `numéro`, au moment où les écrans s'affichent |
| 9 | `SCREEN` | `écran`, `largeur`, `hauteur` (u32), puis `largeur × hauteur × 4` octets RGBA : ce que l'écran montrait pendant l'enregistrement, pour vérifier la relecture |

Un lecteur ignore les types qu'il ne connaît pas, grâce à leur taille.

## Relire

Dans l'ordre : appliquer chaque `MEMORY`, exécuter chaque opération, et à
chaque `SCREEN` comparer son image à celle de l'écran demandé (lu à
l'adresse du dernier `FRAMEBUFFER` de cet écran, comme le fait la 3DS).
