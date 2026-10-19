"""Versioned numerical repair; historical pinned kernels remain unchanged."""
import torch

VERSION='ccnm_fp64_norm_dot_direct_repair_v1'


class PreciseEuclidean:
    def __init__(self,chunk=256,repair_chunk=1024):
        if chunk<1 or repair_chunk<1:raise ValueError('positive chunks required')
        self.chunk,self.repair_chunk=chunk,repair_chunk

    @torch.no_grad()
    def block(self,x,y):
        if (x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or
                x.dtype!=torch.float32 or y.dtype!=torch.float32 or x.device!=y.device):
            raise ValueError('matching FP32 arrays required')
        xx,yy=x.double(),y.double()
        norms=xx.square().sum(1)[:,None]+yy.square().sum(1)[None]
        squared=norms-2*(xx@yy.T)
        rr,cc=(squared<=64*torch.finfo(torch.float64).eps*norms).nonzero(as_tuple=True)
        out=squared.clamp_min_(0).sqrt_()
        for start in range(0,len(rr),self.repair_chunk):
            r,c=rr[start:start+self.repair_chunk],cc[start:start+self.repair_chunk]
            out[r,c]=(xx[r]-yy[c]).square().sum(1).sqrt()
        return out.float()

    def tiles(self,x,y):
        for i in range(0,len(x),self.chunk):
            for j in range(0,len(y),self.chunk):
                yield i,j,self.block(x[i:i+self.chunk],y[j:j+self.chunk])

    def matrix(self,x,y):
        out=torch.empty((len(x),len(y)),device=x.device,dtype=torch.float32)
        for i,j,d in self.tiles(x,y):out[i:i+len(d),j:j+d.shape[1]]=d
        return out

    def nearest(self,x,y):
        if not len(y):raise ValueError('empty support')
        best=torch.full((len(x),),torch.inf,device=x.device)
        ids=torch.full((len(x),),-1,dtype=torch.int64,device=x.device)
        for i,j,d in self.tiles(x,y):
            value,local=d.min(1);target=slice(i,i+len(value));better=value<best[target]
            ids[target]=torch.where(better,j+local,ids[target]);best[target]=torch.minimum(best[target],value)
        return best,ids


def isolated_radii(bank,kernel):
    if len(bank)<2:return bank.new_zeros(len(bank))
    distances=kernel.matrix(bank,bank);distances.fill_diagonal_(torch.inf)
    return distances.min(1).values*.5
