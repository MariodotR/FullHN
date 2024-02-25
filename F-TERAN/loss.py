import torch
from torch import nn as nn
from torch.nn import functional as F
from .utils import l2norm
import numpy as np

def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im)
    s = l2norm(s)
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class Contrastive(nn.Module):
    def __init__(self, margin=0, measure=False, max_violation=False):
        super(Contrastive, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def compute_contrastive_loss(self, scores):
        """
        TRIPLET LOSS
        Parameters: 
        ----------
        scores : corresponde a compute image-sentence score matrix
        TYPE llegan de 30x30 (30=batch size bs) 
        rows captions, columns images
            DESCRIPTION. 
        

        Returns
        -------
        TYPE
            calcula la loss dada similaridades. Calcule el HN = argmax Loss y no HN= argmax=s.
            Yo quiero cambiar la loss pero no el HN. Por lo tanto, debo modificar la loss y que el
            HN se calcule como HN= argmax=s.
        
        """
        #HN_s,HN_im=0,0
        diagonal = scores.diag().view(scores.size(0), 1)  #diagonal a vector columna [s_11,s_22,...,s_bsbs]
        d1 = diagonal.expand_as(scores) # lo deja como matriz repitiendo: [[s_11,..,s_bs, ..., [s_11,..,s_bsbs]]
        d2 = diagonal.t().expand_as(scores) # traspuesta

        # compare every diagonal score to scores in its column
        # caption retrieval
        #[[alpha, alpha + s_12 - s_11, ...., alpha + s_1_bs - s_bsbs],...,[alpha+s_bs1-s_11, ... , alpha] ]
        cost_s = (self.margin + scores - d1).clamp(min=0)  #HINGE LOSS
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        
        HN_s= torch.argmax(scores - d1,  dim=1) #hn de los captions, es decir 
        HN_im= torch.argmax(scores - d2, dim=0)#hn de las imgs, es decir cap
        list_= list(range(scores.size(0)))
        cost_s= cost_s[list_,HN_s].sum()
        cost_im= cost_im[HN_im,list_].sum()
          
        """
        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        #[[0, alpha + s_12 - s_11, ..], [alpha + s_21-s11,...],...,[alpha+s_bs1-s11,...] ]
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation: #HARD NEGATIVE
        #get with argmax wich are those HN
            if return_HN:     
                HN_s= torch.argmax(cost_s,  dim=1) #para cada imagen hay un caption hn
                HN_im= torch.argmax(cost_im, dim=0)
                print("HN L:",HN_s,HN_im )
            #gt_scores=[ scores[i, HN_s[i]]] for i in range(scores.size()[0])
            #maximo error define define la loss pues sería el hn. 
            cost_s = cost_s.max(1)[0] #INPUT l' -> S_ll'
            cost_im = cost_im.max(0)[0] #INPUT k' -> S_kk'
        
        return cost_s.sum() + cost_im.sum(),  HN_s, HN_im
        """
        return cost_s , cost_im,  HN_s, HN_im
 
def intra_similarity(im_set, s_seq, im_len, s_len, mode=False):
        """
        
        print("HABER: ", len(im_len), im_set.size(), len(s_len), s_seq.size() )
        HABER:  30 torch.Size([30, 37, 1024]) 30 torch.Size([30, 31, 1024])
        37,31 son los largos maximos.
        
       """
       # do not consider cls and eos tokens
        if mode== "i2i":
           im_set = im_set[:, 1:, :]
           s_seq = s_seq[:, 1:, :]
           im_len = [l - 1 for l in im_len]
           s_len = [l - 1 for l in s_len]
           
           im_set_batch = im_set.size(0)
           im_set_len = im_set.size(1)
           s_seq_batch = s_seq.size(0) #HN
           s_seq_len = s_seq.size(1) #HN
          
           
           im_set = im_set.unsqueeze(1).expand(-1, s_seq_batch, -1, -1)  # equal to im_set.unsqueeze(0).expand(s_seq_batch, -1, -1, -1)  # B x B x S_im x dim
           s_seq = s_seq.unsqueeze(0).expand(im_set_batch,-1, -1, -1) #s_seq.unsqueeze(0).expand(im_set_batch, -1, -1, -1) # B x B x S_s x dim
           #es necesario para poder multiplicar
           alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))  # B x B x S_im x S_s
              
           #elimine la parte que maneja diferentes largos porque las imagenes son siempre 36 ROI
           return alignments.max(2)[0].sum(2) #Mr_Sw max in img(HN seq) and sum in seq
    
        if mode=="t2t":

            im_set = im_set[:, 1:-2, :]
            s_seq = s_seq[:, 1:-2, :]
            im_len = [l - 3 for l in im_len]
            s_len = [l - 3 for l in s_len]
            
            im_set_batch = im_set.size(0)
            im_set_len = im_set.size(1)
            s_seq_batch = s_seq.size(0)
            s_seq_len = s_seq.size(1)
            #print(im_set.size(), s_seq.size())
            
            im_set = im_set.unsqueeze(1).expand(-1,s_seq_batch, -1, -1)  # equal to im_set.unsqueeze(0).expand(s_seq_batch, -1, -1, -1)  # B x B x S_im x dim
            s_seq = s_seq.unsqueeze(0).expand(im_set_batch, -1, -1, -1) # B x B x S_s x dim
            alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))  # B x B x S_im x S_s
            #print(im_set.size(), s_seq.size(), alignments.size())
            #print(alignments)
            # alignments = F.relu(alignments)

            # compute mask for the alignments tensor for manage diferents length among batchsize is not necesary.
            im_len_mask = torch.zeros(im_set_batch, im_set_len).bool() #BxS_im with false
            im_len_mask = im_len_mask.to(s_seq.device)
            for im, l in zip(im_len_mask, im_len):
                im[l:] = True #length (last 3 or 1 rows ) to end TRUE
            im_len_mask = im_len_mask.unsqueeze(2).unsqueeze(1).expand(-1, s_seq_batch, -1, s_seq_len)
            #im_len_mask = im_len_mask.unsqueeze(1).unsqueeze(0).expand(s_seq_batch, -1, s_seq_len, -1)
            
            s_len_mask = torch.zeros(s_seq_batch, s_seq_len).bool() #BxS_s with false
            s_len_mask = s_len_mask.to(im_set.device)
            for sm, l in zip(s_len_mask, s_len):
                sm[l:] = True #length (last 3 or 1 columns ) to end TRUE
            #s_len_mask = s_len_mask.unsqueeze(2).unsqueeze(1).expand(-1,im_set_batch, -1, im_set_len)
            s_len_mask = s_len_mask.unsqueeze(1).unsqueeze(0).expand(im_set_batch, -1, im_set_len, -1)
            
            alignment_mask = im_len_mask | s_len_mask #columns and rows to TRUE
            #print("t2t mask: ",alignment_mask.sum())
            alignments.masked_fill_(alignment_mask, value=0) # to 0.
            #print(alignments)
            return alignments.max(2)[0].sum(2) + alignments.max(3)[0].sum(2)#alignments.mean(dim=(2,3)) #alignments.max(2)[0].sum(2) #Mr_Sw max in img(HN seq) and sum in seq
            
        

        

        



class AlignmentContrastiveLoss(Contrastive):
    """
    Esta es la loss que se utiliza en train.py que a traves de teran.py aplica el entrenamiento
    
    Tambien en train.py se usa en validate que pide aggr_similarity, es decir la matriz de bsxbs con las similaridades (fila seq, col img).en la diagonal el gt.
    
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, aggregation='sum-max-sentences', name_exp="", lambda_=1, return_similarity_mat=False, mode= False):
        super(AlignmentContrastiveLoss, self).__init__(margin, measure, max_violation)
        self.aggregation = aggregation
        self.return_similarity_mat = return_similarity_mat
        self.mode = mode
        self.name_exp=name_exp
        self.lambda_=lambda_

    def forward(self, im_set, s_seq, im_len, s_len):
        if self.mode=="i2i":
            return intra_similarity(im_set, s_seq, im_len, s_len, mode=self.mode)
        if self.mode=="t2t":
            return intra_similarity(im_set, s_seq, im_len, s_len, mode=self.mode)
        # im_set = im_set.permute(1, 0, 2)    # B x S_im x dim
        # s_seq = s_seq.permute(1, 0, 2)     # B x S_s x dim
        
        im_set_original= im_set # for intra modal
        seq_set_original= s_seq # for intra modal
        im_len_original= im_len # for intra modal
        s_len_original= s_len # for intra modal

        # do not consider cls and eos tokens
        im_set = im_set[:, 1:, :] #[bs, 37 roi, 1024=lat dim]
        s_seq = s_seq[:, 1:-2, :] #[bs, 31 max seq, 1024=lat dim]
        # quitando los tokens extras del largo individual
        im_len = [l - 1 for l in im_len] # len = bs
        s_len = [l - 3 for l in s_len] #len = bs
        
        im_set_batch = im_set.size(0) # bs
        im_set_len = im_set.size(1) # max roi = 37 -1
        s_seq_batch = s_seq.size(0) # bs
        s_seq_len = s_seq.size(1) # max seq = 31 -1

        im_set = im_set.unsqueeze(1).expand(-1, s_seq_batch, -1, -1)  # B x B x S_im x dim
        s_seq = s_seq.unsqueeze(0).expand(im_set_batch, -1, -1, -1) # B x B x S_s x dim
        alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))  # B x B x S_im x S_s
        # alignments = F.relu(alignments)

        # compute mask for the alignments tensor
        im_len_mask = torch.zeros(im_set_batch, im_set_len).bool()
        im_len_mask = im_len_mask.to(im_set.device)
        for im, l in zip(im_len_mask, im_len):
            im[l:] = True
        im_len_mask = im_len_mask.unsqueeze(2).unsqueeze(1).expand(-1, s_seq_batch, -1, s_seq_len)

        s_len_mask = torch.zeros(s_seq_batch, s_seq_len).bool()
        s_len_mask = s_len_mask.to(im_set.device)
        for sm, l in zip(s_len_mask, s_len):
            sm[l:] = True
        s_len_mask = s_len_mask.unsqueeze(1).unsqueeze(0).expand(im_set_batch, -1, im_set_len, -1)

        alignment_mask = im_len_mask | s_len_mask
        alignments.masked_fill_(alignment_mask, value=0)
        # alignments = F.relu(alignments)
        # alignments = F.normalize(alignments,p=2, dim=2)

        if self.aggregation == 'sum':
            aggr_similarity = alignments.sum(dim=(2,3))
        elif self.aggregation == 'mean':
            aggr_similarity = alignments.mean(dim=(2,3))
        #--------------------------------------------------  
        elif self.aggregation == 'MrSw': #este es el bueno
            #print("alignments: ", alignments.shape )
            aggr_similarity = alignments.max(2)[0].sum(2) +alignments.max(3)[0].sum(2) #por palabra se suma su max region afín
            #print("matrix__:", aggr_similarity.shape)
        #--------------------------------------------------  
            
        elif self.aggregation == 'MrAVGw':
            aggr_similarity = alignments.max(2)[0].sum(2)
            expanded_len = torch.FloatTensor(s_len).to(alignments.device).unsqueeze(0).expand(len(im_len), -1)
            aggr_similarity /= expanded_len
        elif self.aggregation == 'symm':
            im = alignments.max(2)[0].sum(2)
            s = alignments.max(3)[0].sum(2)
            aggr_similarity = im + s
        elif self.aggregation == 'MwSr':
            
            aggr_similarity = alignments.max(3)[0].sum(2)
            
        elif self.aggregation == 'scan-sentences':
            norm_alignments = F.relu(alignments)
            norm_alignments = F.normalize(norm_alignments,p=2, dim=2)
            weights = norm_alignments.masked_fill(alignment_mask, value=float('-inf'))
            weights = torch.softmax(weights, dim=3)

            weights = weights.unsqueeze(3)  # B x B x im x 1 x s
            s_seq_ext = s_seq.unsqueeze(2).expand(-1, -1, im_set_len, -1, -1)
            att_vector = torch.matmul(weights, s_seq_ext)  # B x B x im x 1 x dim
            att_vector = att_vector.squeeze(3)
            new_alignments = F.cosine_similarity(im_set, att_vector, dim=3)  # B x B x im
            new_alignments.masked_fill_(im_len_mask[:, :, :, 0], value=0)

            aggr_similarity = new_alignments.sum(2)

        if self.return_similarity_mat:
            return aggr_similarity
        else: #modificacion
            loss_i2t, loss_t2i, HN_s, HN_im = self.compute_contrastive_loss(aggr_similarity) 
            gt_scores=  aggr_similarity.diag().view(aggr_similarity.size(0), 1)
            #visual constrain
            #select HN
            im_set_HN= im_set_original[ HN_s,:,:]
            im_len_HN=  [im_len_original[i] for i in HN_s]
            
            compute_sim_i2i= intra_similarity(im_set_original,im_set_HN,im_len_original,im_len_HN, mode="i2i") 
            sim_i2i=compute_sim_i2i.diag().view(compute_sim_i2i.size(0), 1)  
            loss_vc= (self.margin + sim_i2i - gt_scores).clamp(min=0).sum()
            loss_vc= loss_vc  /100 #/(loss_i2t+loss_t2i)
            
            #textual constrain 
            #select HN
            seq_set_HN= seq_set_original[ HN_im,:,:]
            s_len_HN=  [s_len_original[i] for i in HN_im]
            
            compute_sim_t2t= intra_similarity(seq_set_original,seq_set_HN, s_len_original,s_len_HN,mode="t2t") 
            sim_t2t=compute_sim_t2t.diag().view(compute_sim_t2t.size(0), 1)  
            loss_tc= (self.margin + sim_t2t - gt_scores).clamp(min=0).sum()
            loss_tc= loss_tc /10
            
            #if visual and texual HN aren't a positive pair = not in the diagonal.
            loss_ps=0
            for i in range(len(HN_s)):
                if HN_s[i]!=HN_im[i]:
                    loss_ps+= (self.margin + aggr_similarity[HN_im[i],HN_s[i]] - gt_scores[i] ).clamp(min=0) 
            loss_ps= loss_ps[0]    
            
            str_loss_monitoreo= "{i2t} ; {t2i}; {vc}; {tc}; {ps}".format(i2t=loss_i2t, t2i=loss_t2i,vc=loss_vc, tc=loss_tc, ps =loss_ps)
            print("Loss: ", str_loss_monitoreo)
            f = open(self.name_exp+'/loss.txt', 'a')
            f.write( str_loss_monitoreo+"\n")
            f.close()
            return loss_i2t+loss_t2i+ self.lambda_*(loss_vc+loss_tc+loss_ps)


class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        return self.compute_contrastive_loss(scores)


class PermInvMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # @staticmethod
    # def batched_cosine_sim(im, s):
    #     """Cosine similarity between all the image and sentence pairs
    #     """
    #     im = F.normalize(im, p=2, dim=2)
    #     s = F.normalize(s, p=2, dim=2)
    #     return im.mm(s.permute(0, 2, 1))

    def forward(self, im, s):
        dist_matrix = torch.cdist(im, s, p=2)
        row_sum = F.softmin(dist_matrix, dim=2).max(dim=2)[0].sum(dim=1)
        col_sum = F.softmin(dist_matrix, dim=1).max(dim=1)[0].sum(dim=1)
        loss = 2*torch.Tensor([dist_matrix.shape[1]]).to(im.device) - row_sum - col_sum
        loss = loss.mean()
        return loss
