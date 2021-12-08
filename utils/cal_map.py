import numpy as np

def mean_average_precision(query_code,query_label,retrieval_code,retrieval_label,R):
    query_code = query_code.cpu().detach().numpy()
    query_label = query_label.cpu().detach().numpy()
    retrieval_code = retrieval_code.cpu().detach().numpy()
    retrieval_label = retrieval_label.cpu().detach().numpy()
    query_num=query_code.shape[0]
    retrieval_code=np.sign(retrieval_code)
    query_code=np.sign(query_code)

    sim=np.dot(retrieval_code,query_code.T)   #矩阵积
    ids=np.argsort(-sim,axis=0)
    APx=[]

    for i in range(query_num):
        label=query_label[i,:]
        label[label==0]=-1   #这里是为了检测是否有相同标签时，不受0的影响
        idx=ids[:,i]
        imatch=np.sum(retrieval_label[idx[0:R],:]==label,axis=1)>0   #retrieval_data是否和这个query有同样的标签。
        relevant_num=np.sum(imatch)
        Lx=np.cumsum(imatch)
        Px=Lx.astype(float)/np.arange(1,R+1,1)
        if relevant_num !=0:
            APx.append(np.sum(Px*imatch)/relevant_num)
        else:
            APx.append(0.0)

    return np.mean(np.array(APx))

