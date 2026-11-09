# picanative

Moteur de rendu natif pour les ports PC et Android de jeux Nintendo 3DS
décompilés. Il reçoit ce que le code d'un jeu envoie au GPU PICA200 de la
console (listes de commandes, effacements, copies, échanges d'images) et le
dessine avec Vulkan, sur Windows et sur Android.

Premier client : le port natif de Super Mario 3D Land, sur PC et sur l'AYN
Thor.

**État : conception.** L'architecture et les étapes sont dans
[DESIGN.md](DESIGN.md). Aucun module n'est encore écrit.

## Principes

- **Ports d'abord.** Le moteur sert un jeu décompilé et compilé nativement,
  pas un émulateur. Il remplace le service GPU de la console au niveau de l'API
  du SDK.
- **Modulaire.** Des modules C++20 qui ne communiquent que par des données :
  décodage de l'état PICA, génération de shaders, textures, surfaces, une petite
  interface de rendu et son implémentation Vulkan.
- **Rapide.** SPIR-V produit directement, caches de shaders et de pipelines,
  aucun retour du GPU vers le processeur sans nécessité, résolution interne
  réglable.
- **Vérifié.** Chaque étape est comparée pixel par pixel, sur des traces
  enregistrées, à une implémentation de référence.

## Ce que ce dépôt ne contient pas

Ni ROM, ni code, ni données, ni traces tirées d'un jeu. Les traces servent aux
tests mais restent chez celui qui les enregistre, à partir de sa propre copie.

## Licence

MIT, voir [LICENSE](LICENSE).
