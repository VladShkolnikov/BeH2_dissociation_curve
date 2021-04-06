
import vqe_methods_add_by_one_Harper_truncation
import pickle
import sys



r=sys.argv[1]

PoolOfHope=['ZYXIZZZZZYYI', 'YXIIZZIIYYII', 'ZIXYZZZIYYII', 'XXIZZZYXIIII', 'XYZIZIYYZIII', 'IIYXYYZZZZII', 'ZZYXIZYYIIII', 'YZIXZZZIIYYI', 'IXXZIIIZZXYI', 'XIIIZZXYYIXY', 'XXXZYXXXYXYI', 'ZXIIIZZZZYII', 'XIZZIZXYZXII', 'YZXZZIZZYZYI', 'YZYXZIZIXZXY', 'ZZZIZIIZXXXY', 'IZZZYYYXYXXY']
Resultat=[]


geometry = "H 0 0 0; Be 0 0 {}; H 0 0 {}".format(r,2*r)
print(geometry)
vqe_methods_add_by_one_Harper_truncation.adapt_vqe(geometry,
	                  adapt_thresh    = 1e-8,                        #gradient threshold
                      adapt_maxiter   = 400,                       #maximum number of ops                   
                      Pool            = PoolOfHope,
                      Resultat        = Resultat,
                      bond_legth     = r
                      ) 
    
with open('Bond_length_dependence.BeH2_dissociation_curve_pickle_min_pool_{}'.format(r) , 'wb') as handle:
    pickle.dump(Resultat, handle, protocol=pickle.HIGHEST_PROTOCOL)                        





