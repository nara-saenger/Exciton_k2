import random 
import math
import matplotlib.pyplot as plt
import numpy as np

# Funções auxiliares 

def calc_dt(k):
    u = random.random()
    return -np.log(u)/sum(k)

def K_to_prob(k):
    K_T = sum(k)
    return [k[i]/K_T for i in range(len(k))]

def which_event(p):
    s = [sum(p[:i+1]) for i in range(len(p))]
    u = random.random()
    if u < s[0]:
        return 0
    if u < s[1]:
        return 1
    if u <= s[2]:
        return 2 

def calc_kappa2(mu_d, mu_a):
    mu_d = np.array(mu_d)
    mu_a = np.array(mu_a)
    kappa = np.dot(mu_d,mu_a) - 3*(mu_d[0])*(mu_a[0])  # Rede na coordenada x
    return kappa**2

def calc_K(kappa2,R_0,r,Tau):
    K = (((R_0/r)**6)*kappa2)/Tau
    return K
    
def cond_contorno(N,X_0):
    if X_0 > N:
        return 0
    if X_0 < 0:
        return N
    return X_0


def traj(d,R_0,r,Tau):
    N = len(d)     #Tamanho da rede
    X_0 = random.randint(0,N-1)
    alive = True
    t = 0
    n = 0
    traj = [X_0]
    x = X_0
    x_rel = 0 #distancia relativa
    contagem = 0
    cont_esq = 0
    cont_dir = 0
    

    while alive:
        x_E_con = cond_contorno(N-1,X_0 -1)
        x_D_con = cond_contorno(N-1,X_0 +1)
        s_i = d[X_0]
        s_E = d[x_E_con]
        s_D = d[x_D_con]
        kappa2_E = calc_kappa2(s_i,s_E)
        kappa2_D = calc_kappa2(s_i,s_D)
        
        print(kappa2_E,kappa2_D)
        K_Emi = 1/Tau
        K_Es = calc_K(kappa2_E,R_0,r,Tau)
        K_Dir = calc_K(kappa2_D,R_0,r,Tau)

        list_K = [K_Emi, K_Dir, K_Es]

        P = K_to_prob(list_K) 
        dt = calc_dt(list_K)
        t += dt
        event = which_event(P)

        if event ==0:
            alive = False
        if event == 1:    #Pulo para a direita
            X_0 = x_D_con
            x += 1
            contagem +=1
            cont_dir +=1
            x_rel += r
            traj.append(x)
        if event ==2:     #Pulo para a esquerda 
            X_0 = x_E_con
            x -= 1 
            contagem +=1
            cont_esq += 1
            x_rel -= r
            traj.append(x)

    return(traj, t ,contagem, cont_dir,cont_esq, x_rel)

###################################
#Função que calcula o LD

def L_D_2(d,R_0,r,Tau,n):  #Pulos para os dois lados modificado
    s_N = []
    s = []
    for i in range(n):
        t = traj(d,R_0,r,Tau)
        s_N.append(t[2]) 
        delta_x = t[5]
        s.append(delta_x)
    
    passos = (sum(s_N)/n)
    L_D_steps = r*(passos**0.5)
    L_D_sig = np.sqrt(np.mean([x**2 for x in s]))

    RF = ((2/3)**(1/6))*R_0
    L_D_an = (np.sqrt(2)*(RF)**3)/(r**2)


    return L_D_an, L_D_steps, L_D_sig

###################################
#Rede aleatória 

def rede(N):
    lista = []
    for i in range(int(N)):
        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1,1)
        z = np.random.uniform(-1,1)
        norm = (x**2 + y**2 + z**2)**0.5
        x = x/norm
        y = y/norm
        z = z/norm
        lista.append([round(x,5),round(y,5),round(z,5)])
        
    return lista

N_rede = 100   #(tamanho da rede)
d = rede(N_rede)

#
# Comprimento de difusão (variando R_0)

from joblib import Parallel, delayed
R_0 = 1    # R_F = k^2 * R_0
r = 1      # distância entre as moléculas 
Tau = 100
r = 1
n = 1000                       #número de éxcitons 
R_0 = np.linspace(0.1,4.0,30)     # variação de R_0

resultados = Parallel(n_jobs=-1,verbose=10)(delayed(L_D_2)(d,R_0[i],r,Tau,n) for i in range(len(R_0)))

L_D_sig = [resultados[i][2] for i in range(len(R_0))]
L_D_steps = [resultados[i][1] for i in range(len(R_0))]
L_D_ana = [resultados[i][0] for i in range(len(R_0))]

# Gráfico do comprimento analítico vs calculado pelo número de passos

lista_L_D_an = L_D_ana

plt.figure()
plt.plot(L_D_steps, lista_L_D_an, "o")
plt.xlabel(r"$L_D$ steps")
plt.ylabel(r"$L_D$ 2/3")
L_D_max = max(L_D_steps + lista_L_D_an)
plt.plot([0, L_D_max], [0, L_D_max], '--', label = 'Curva ideal')
plt.legend()
plt.savefig('test.png')
