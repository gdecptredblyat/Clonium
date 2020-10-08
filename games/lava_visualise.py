import random
class Item:
    def __init__(self,i, j, deep):
        self.i = i
        self.j = j
        self.deep = deep

class Gamer:
    def __init__(self, color, side, i, j):
        self.color = color
        self.side = side
        self.i = i
        self.j = j


class CField:
    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.items = []
        for i in range(length):
            self.items.append([1] * width)
        for i in range(length):
            for j in range(width):
                self.items[i][j] = Item(i, j, random.randint(1, 3))


def Print_Field():
    for f in range(length):
        st = []
        for g in range(width):
            st.append(Field.items[f][g].deep)
        print(st)
    print("Position gamer_1: ", Gamer_1.i+1,Gamer_1.j+1,"Depth position gamer_1: ", Field.items[Gamer_1.i][Gamer_1.j].depth," . Position gamer_2: ",Gamer_2.i+1,Gamer_2.j+1,". Depth position gamer_2: ", Field.items[Gamer_2.i][Gamer_2.j].depth )

def Move(i,j):
    G_1.side = definition_side(i,j)
    for k in range(len(Aviable_sides)):
        if (G_1.side == Aviable_sides(k)):
            if(i==G_2.i) and (j==G_2.j):
                attack(i,j)
            if (Field.items[G_2.i][G_2.j].deep == 0):
                Game = 0
            else:
                aviable_sides(G_1.i, G_1.j)
                aviable_sides(G_2.i, G_2.j)

    aviable_sides(G_1.i,G_1.j)





def attack(i,j):
        #  if (G_2.i==length-1)and   если она находится на границе
        G_1.i=G_2.i
        G_1.j=G_2.j
        G_1.side = definition_side(i, j)
        # прибавляем г_2 +1 в и либо ж
        if G_1.side == "up":
            G_2.i+=1
        if G_1.side == "down":
            G_2.i -= 1
        if G_1.side == "left":
            G_2.j -= 1
        if G_1.side == "right":
            G_2.j += 1





def definition_side(G_1, i,j):
    if (G_1.i < i) and (G_1.j == j):
        return "up"
    if (G_1.i == i) and (G_1.j < j):
        return "right"
    if (G_1.i > i) and (G_1.j == j):
        return "down"
    if (G_1.i == i) and (G_1.j > j):
        return"left"
    else:
        return "yourself"



def aviable_sides(i,j):
    if (G_1.i+1==length):
    #запоминать все прошлые ходы и если он нажал сам на себя то давать право ходить в любую сторону
        G_1.side = definition_side(i,j)
    if G_1.side == "yourself":
        Aviable_sides=All_sides
    else:
        Aviable_sides.remove(G_1.side)


length = int(input("Enter length: "))
width = int(input("Enter width: "))
Field = CField(length, width)
# Main_DList = DList_1
Position_error=True
while Position_error:
    Gamer_1 = Gamer(1, "yourself", random.randint(0,length),random.randint(0,width))
    Gamer_2 = Gamer(2, "yourself",random.randint(0,length),random.randint(0,width))
    if (Gamer_1.i==Gamer_2.i) and (Gamer_1.j==Gamer_2.j):
        Position_error=True
    else:
        Position_error=False

first_player_error=True
second_player_error=True
Game = 1
wave=1

num_gamers = 2

Game_over = [[0]*num_gamers]


Aviable_sides_1=["up", "right", "down", "left", "yourself"]
Aviable_sides_2=["up", "right", "down", "left", "yourself"]
All_sides=["up", "right", "down", "left", "yourself"]

while Game:
    for gamer in range(num_gamers):
        G_1 = Gamers[gamer]
        while player_error:
            i_1 = int(input("Enter i_1: "))
            j_1 = int(input("Enter j_1: "))
            def_side = definition_side
            if (def_side in Aviable_sides_arr[gamer]) and (Field.items[i_1][j_1].deep>0):
                Field.items[i_1][j_1].deep -= 1
                if def_side == 'yourself':
                    if Field.items[i_1][j_1].deep == 0:
                        print('Game_over for '+ (gamer+1))
                        Game_over[gamer] = 1
                        continue
                    Aviable_sides_arr[gamer] = All_sides
                else: 
                    Aviable_sides_arr[gamer].remove(def_side)
                player_error = False
            else:
                print('AGAIN')

    if Game_over.all == 1 and (wave > 1):
        Game = 0
        print('Game is over')
        # if (G_1.sum == 0):
        #     print("Game over. Gamer_2 win!!!")
        # else:
        #     print("Game over. Gamer_1 win!!!")
    
    wave +=1
