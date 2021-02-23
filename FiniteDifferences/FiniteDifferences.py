import sys
import math
import pandas as pd 
import numpy as np





####################################################################################################################
machine_epsilon = sys.float_info.epsilon
####################################################################################################################
###########################            FUNCTION DECLARATIONS          ##############################################
####################################################################################################################
def kronecker( i : int , j : int ) -> int :
    #
    result = 1 - math.ceil( abs(i-j)/(1+abs(i)+abs(j)))
    #
    return result
#
#
#
def derappnum(f_x : list , d : int ,  h : float = None , p : int = 1) -> list:
    
    #if d >= 1 and p >= 1:
    #M für Forward/Backard und centered def 
    
    M   = int (d + p -1)
    M_c = math.floor(0.5 * M)
        
    #Vorfaktor d!/(h**d) def. 
    fact = math.factorial(d) / (h ** d)
#############################################################################################
    # Forward:
    for index in range(M_c) :

        #i_min und i_max festlegen

        i_min_f = int(0)
        i_max_f = int(M)
        
        
        #Vektor B mithilfe Kronecker Delta: 
        B_f = pd.Series(index=range(M + 1), dtype = "int")
        B_f = B_f.fillna(0) # with 0s rather than NaNs
        #print("B_f = \n",B_f)



    
        for n in range(M + 1): 
            
            B_f[n] = kronecker (n, d) + (1 -  kronecker (n, d)) * (math.floor(n/M) -  kronecker (n, M))
            
            #print ("B_f[n]=",B_f[n])
            #print("\nn=", n,"\ndelta(",n,",",d,")=", kronecker(n,d))
            #print("ausdruck = ", (kronecker(n,d)+(1-kronecker(n,d)) * (math.floor(n/M) -  kronecker (n, M))))
            
                   
    #print ("B_f=\n",B_f)
        
    #Matrix C: 
        
    n_rows = int (M + 1)
    i_cols = int (M + 1)
                
    C_f = pd.DataFrame(index = range(n_rows), columns = range(i_cols), dtype='float')
        
    for n in range (M + 1) : 
            
                for i in range (i_min_f, i_max_f+1):
                
                     C_f.iloc[n,i] = i**n
            
    #print("\n\nC_f=\n", C_f,"\n\n")      
##############################################################################################
    #Centered: 
    for index in range (M_c, len(f_x) - M_c) :
        
        #imin und imax festlegen: 
        
        i_min_c = int(-M_c)
        i_max_c = int(M_c)
        
        

        #Vektor B mithilfe Kronecker Delta: 
        B_c = pd.Series(index=range(M + 1), dtype = "int")
        B_c = B_c.fillna(0) # with 0s rather than NaNs
      

        for n in range(M+1): 
            
            B_c[n] = kronecker (n, d) + (1 -  kronecker (n, d)) * (math.floor(n/M) -  kronecker (n, M))


    #print ("\n\nB_c=\n",B_c,)
    #print("\nShape of B_c", B_c.shape)
        
    #Matrix C: 
        
    i_cols = 2*M_c + 1
    n_rows = M + 1
    #print("\n\ni_rows = ",n_rows)                
    #print("n_cols = ",i_cols)                
    C_c = pd.DataFrame(index = range(i_cols), columns = range(n_rows),dtype='float')

    
    #print("Shape of C = ",C_c.shape)
    for n in range (M + 1) : 
            
                for i in range (i_min_c, (i_max_c+1), 1):
                    #print("\nn =,", n)
                    #print("    i=", i)
                    index = i + M_c
                    value = i**n
                    C_c.iloc[n,index] = value
                    
            

    #print("\n\nC_c=\n", C_c,"\n\n")
##############################################################################################
 # Backward:
    for index in range((len(f_x) - M_c) , len(f_x)) :

        #i_min und i_max festlegen

        i_min_b = -M
        i_max_b = 0
        
        
        #Vektor B mithilfe Kronecker Delta: 
        B_b = pd.Series(index=range(M+1), dtype = "int")
        B_b = B_b.fillna(0) # with 0s rather than NaN


        for n in range(M + 1): 
            
            B_b[n] = kronecker (n, d) + (1 -  kronecker (n, d)) * (math.floor(n/M) -  kronecker (n, M))
            
            
    #print ("B_b=\n",B_b)
        
    #Matrix C: 
        
    n_rows = int (M + 1)
    i_cols = int (M + 1)
                
    C_b = pd.DataFrame(index = range(n_rows), columns = range(i_cols), dtype='float')
        
    for n in range (M + 1) : 
            
                for i in range (i_min_b, (i_max_b+1), 1):
                    #print("\nn =,", n)
                    #print("    i=", i)
                    index = i + M
                    #print("      index=", index)
                    value = i**n
                    C_b.iloc[n, index] = value
                    
    #print("\n\nC_b=\n", C_b)  
##################################################################################################
    #SVD für alle 3 Intervalle berechnen: 
    #Forward:
    u_f, s_f, v_f = np.linalg.svd(C_f)
    #Test für Forward: 
    #print('\n\nTest SVD Forward:\n',u_f, '*\n', s_f, '*\n',v_f, '=\n',C_f) 
    #print('\n\n Rekonstruktion:', np.allclose(C_f, (u_f * s_f) @ v_f))
    #print("\n\nDimensionen:", u_f.shape, s_f.shape, v_f.shape)
    #Centered
    u_c, s_c, v_c = np.linalg.svd(C_c)
    #Backward:
    u_b, s_b, v_b = np.linalg.svd(C_b)

    #Fall, dass ein Eigenwert = 0 in s für alle 3 Methoden ausschließen (für i_min <= i <= i_max) 
    #### 
    s_ges = list()
    s_ges = np.concatenate([s_f, s_c, s_b])
     #print("Liste aller Eigenwerte:", s_ges)
    #print("Länge:", len(s_ges))
    reviser = int (0)
    if s_ges[reviser] != float(0):
####################################################################################################
        # jeweils die Inverse von C berechnen...diese ergibt sich bei U * s *VT zu V s^-1 *UT (Quelle...)

        C_f_i = pd.DataFrame(np.transpose(v_f) @ np.diag(1/s_f) @ np.transpose(u_f))
        C_c_i = pd.DataFrame(np.transpose(v_c) @ np.diag(1/s_c) @ np.transpose(u_c))
        C_b_i = pd.DataFrame(np.transpose(v_b) @ np.diag(1/s_b) @ np.transpose(u_b))
    
        #Test: Ausgabe: 
        #print("C_c_i =", C_c_i)
    
        #Test: Eiheitsmatrix? 
        ident_mat = pd.DataFrame(C_c_i @  C_c, index = range(n_rows), columns = range(i_cols))
    
        #print('\n\n Test: Einheitsmatrix =', ident_mat)

        
####################################################################################################

        #Koeffizienten A für alle 3 Intervalle: 
        A_f = C_f_i @ B_f
        A_c = C_c_i @ B_c
        A_b = C_b_i @ B_b
    
    
        #Test 
        #print("\n\nA_f=", A_f)
        
##################################################################################################### 
        #Ableitungswerte in allen 3 Intervallen berechnen 
        print("\n\n")
        #Array für Ableitung 
        f = list()
    
        #Forward: 
    
        for a in range (M_c): 
        
            arr = list()
        
            for i in range (i_min_f, (i_max_f+1), 1):
            
                arr.append(A_f[i]*f_x[a + i])
                #print("a =", a)
                #print("    i =", i)
            f.append(fact*math.fsum(arr))
        
        #Centered
        for a in range(M_c, len(f_x) - M_c): 
        
            arr.clear()
            arr = list()

        
            for i in range (i_min_c, (i_max_c+1), 1):
        
                index = M_c + i
            
                arr.append(A_c[index]*f_x[a + i] ) 
            
                #print("a =", a)
                #print("  i =", i)
                #print("  index=",index)
            f.append(fact*math.fsum(arr))   
        
        #Backward 
        for a in range((len(f_x) - M_c) , len(f_x)): 
        
            arr.clear()
            arr = list()

        
            for i in range (i_min_b, (i_max_b+1), 1):
        
                index = M + i
            
                arr.append(A_c[index]*f_x[a + i] ) 
            
                #print("a =", a)
                #print("    i =", i)  
                #print("      index", index)
            f.append(fact*math.fsum(arr))   
                        
        #Ausgabe der Liste mit den entgültigen Ableitungswerten: 
        #print("Ableitungswert:", f)
        print("-------Ableitung erfolgreich berechnet------------")
        print("\nLänge Ausgangsliste:",len(f_x))
        print("\nLänge Liste Ableitungswerte:", len(f))
    else: 
        print("!!!Einer der Eigenwerte ist 0!!!")
        
    return f
