# PREVIEW from the real insertion_model.py output pasted on 2026-10-27 (NO pressure correction; dg0-slope covariance not available -> treated as independent)
import sys, numpy as np
sys.path.insert(0, "mlp_pipeline/tdi"); import solid_hull as sh
T = [(300,5000,-324.361,29.389,49.304,9.584),(400,5000,-480.592,51.236,51.962,14.355),(400,6500,-306.147,41.047,25.620,12.132),
     (500,5500,-502.686,38.413,25.741,12.175),(500,6000,-554.091,29.459,44.575,9.518),(500,6500,-548.031,41.969,44.131,13.268),(500,7000,-402.889,38.238,18.764,11.439)]
cryst = {(500,5500):141.0, (500,6000):76.0, (400,5000):192.0}   # H2.25O direct, meV per added H (CURRENT_ISSUES); errors not at hand -> drawn hollow, not fitted
st=[]
for P,Tk,d,de,s,se in T:
    c=[dict(phase="fcc",n=27,nO=108,avg=cryst[(P,Tk)],err=0.0,caveat=True,why="value only")] if (P,Tk) in cryst else []
    st.append(dict(P=P,T=Tk,nO=108,ins=[],cryst=c,fits={"ins":(np.array([d,s]),np.diag([de**2,se**2]))}))
sh.plot(st,"docs/figures/solid_hull_preview",0.1,"fcc",note="PREVIEW from insertion_model fits (no P correction, fit covariance unknown: band too wide); crystal H2.25O values without error bars. Full version: solid_hull.py on summary.csv")
