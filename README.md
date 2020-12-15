# Peut-on prédire la note d'une série sur Sens Critique ?

L'objectif de ce projet est de déterminer dans quelle mesure les critères généraux d'une série (période de diffusions, durée, genres...) peuvent aider à prédire sa note.

# Les données:

Sens Critique est un site français spécialisé de critique de livres, films et séries. Sur la page de présentation d'une série figurent notamment la note moyenne accordées par les utilisateurs et des critères généraux (Auteurs, période de diffusions, plateforme, genres, ect...).
Chaque année le site publie un classement des 100 (jusqu'à 2013) puis 50 (depuis 2014) meilleures séries de l'année, qui ne se base pas sur la note moyenne mais sur un vote des utilisateurs. Ainsi, ce classement contitue une source assez fiable de séries ayant été visionné par un nombre minimal de personnes, et dont l'échelle des notes est assez étalée, puisque la note moyenne n'est pas prise en compte (même si la distribution des notes reste centrée sur des notes assez élevées par rapport à la majorité des séries).

# Les fichiers
Le notebook wescrapping contient le code de l'extraction des données de SensCritique en allant chercher le nom des séries dans le classement, puis les informations relatives à celles-ci sur leur page.

Le notebook traitement_data s'occupe de la transformation des informations extraites, pour qu'on puisse les utiliser dans les étapes suivantes

Dans projet_visualisation, on s'intéresse à décrire le jeu de donnée selon les informations que l'on a extraites de SensCritique. On regarde ainsi, s'il en exite un dont les variations de valeurs sont discriminantes pour la note.

Dans régression_linéaire, on appliquera un modèle de régression linéaire au jeu de donnée, ce qui permettra de déterminer l'impact des différents critères sur la note.
On retrouve également le jeu de donnée après extraction (data), traitement (data_final) et transformation pour la régression (data_reg).
