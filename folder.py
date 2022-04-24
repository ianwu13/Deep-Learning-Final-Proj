import os
destination = r'C:\Users\wierd\Documents\GitHub\Deep-Learning-Final-Proj\Dataset\Anime/'

anime = ['Black Butler', 'Black Cat', 'Black Lagoon', 'Bleach', 'Btooom', 'Chobits', 'Code Geass', 'Cowboy Bebop', 'Darker Than Black', 'Deadman Wonderland', 'Death Note', 'Demon Slayer', 'Dragon ball', 'Dragon Ball Z Kai', 'Ergo Proxy', 'Eureka Seven', 'Fairy Tail', 'Fighting Spirit', 'Food Wars', 'Fullmetal Alchemist Brotherhood', 'Future Diary', 'Gekijouban FateStay Night Unlimited Blade Works', 'Get Backers', 'Haikyuu', 'Hunter x Hunter', 'Kuroko Basketball', 'Magi The labyrinth of magic', 'Monster', 'Naruto', 'Naruto Shippuden', 'Neon Genesis Evangelion', 'No Game, No Life', 'One Outs', 'One Piece', 'one punch man', 'Paranoia Agent', 'Parasyte The Maxim', 'Phantom Requiem for the Phantom', 'Psycho Pass', 'Puella Magi Madoka Magica', 'Rurouni Kenshin', 'Steins Gate', 'The Seven Deadly Sins', 'Tokyo Ghoul', 'Your Lie in April']

source = r'C:\Users\wierd\Documents\GitHub\Deep-Learning-Final-Proj\Dataset\Anime/'

for a in anime:
    newSource = source + a + '/'
    allfiles = os.listdir(newSource)
    for f in allfiles:
        os.rename(newSource + f, destination + a + f)
