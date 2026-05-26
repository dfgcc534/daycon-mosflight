import sys, importlib.util as u
import numpy as np
sys.path.insert(0,'.')
from src.io import load_all_samples
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
def L(p,n): s=u.spec_from_file_location(n,p); m=u.module_from_spec(s); s.loader.exec_module(m); return m
kf=L('analysis/plan-a-002/kalman_features.py','kf'); yw=L('analysis/plan-a-001/yaw.py','yw')
R=0.01
ids,X=load_all_samples('train'); n=len(ids); dt=0.04
z8=np.load('analysis/plan-a-002/results_kr008.npz'); y=z8['y']; kal=z8['kalman_main']; fid=z8['fold_ids']; pred8=z8['oof_pred']
def hit(p): return (np.linalg.norm(p-y,axis=-1)<=R)
theta=yw.yaw_from_last_step(X)
r_yaw=yw.rotate_xy(y-kal,theta)
v=(X[:,-1]-X[:,-2])/dt; a=(X[:,-1]-2*X[:,-2]+X[:,-3])/dt**2; cc=kf.cv_ca_disagreement(X)
rel3=(X[:,-3:]-X[:,-1:]).reshape(n,-1)
feats=np.concatenate([v,a,cc,rel3,np.linalg.norm(v,axis=1,keepdims=True),np.linalg.norm(cc,axis=1,keepdims=True)],axis=1)
ar=np.arange
print(f"KR008 OOF hit={hit(pred8).mean():.4f}\n")
print("K | oracle@K | bestmode GBM_acc/chance(Δ) | 실현hit(예측모드) | 실현hit(KR008+alt)")
for K in [2,3,4,6,8]:
    cand=np.zeros((n,K,3)); bestk=np.zeros(n,int); pk_oof=np.zeros(n,int)
    real=np.zeros((n,3)); real_h=np.zeros((n,3)); orac=np.zeros(n,bool)
    for f in range(5):
        tr=fid!=f; va=fid==f
        C=KMeans(K,n_init=5,random_state=0).fit(r_yaw[tr]).cluster_centers_
        cw=np.stack([kal+yw.inverse_rotate_xy(np.tile(C[k],(n,1)),theta) for k in range(K)],axis=1) # (n,K,3)
        d=np.linalg.norm(cw-y[:,None,:],axis=-1); bk=d.argmin(1)
        clf=GradientBoostingClassifier(max_depth=3,n_estimators=120,subsample=0.7).fit(feats[tr],bk[tr])
        pk=clf.predict(feats[va])
        cand[va]=cw[va]; bestk[va]=bk[va]; pk_oof[va]=pk; orac[va]=d[va].min(1)<=R
        real[va]=cw[va][ar(va.sum()),pk]
        hyb=cw[va].copy(); hyb[:,0]=pred8[va]; real_h[va]=hyb[ar(va.sum()),pk]
    acc=(pk_oof==bestk).mean(); chance=np.bincount(bestk,minlength=K).max()/n
    print(f"{K} | {orac.mean():.4f} | {acc:.4f}/{chance:.4f}(Δ{acc-chance:+.3f}) | {hit(real).mean():.4f} | {hit(real_h).mean():.4f}")
print("\n해석: 실현hit(KR008+alt) > 0.6671 면 user 아이디어 작동. GBM_acc≈chance 면 모드도 예측불가(천장 재확인).")
print("MODE_SEL_DONE")
