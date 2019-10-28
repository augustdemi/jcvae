import torch
from torch.autograd import Variable
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.normal import Normal


def elbo(q, p, rec, latents=None, sample_dim=None, batch_dim=None, lamb=1.0, beta=[1.0,1.0,1.0],
         size_average=True, bias=None):
    reconst_loss = rec.loss

    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_probability(q, p, latents, sample_dim, batch_dim, bias)
    kl = beta[0] * (log_q_zCx - log_qz) +  beta[1] * (log_qz - log_prod_qzi) + beta[2] * (log_prod_qzi - log_pz)

    elbo = lamb * reconst_loss - kl
    elbo = elbo.mean(sample_dim) # across sample_dim
    elbo = elbo.mean() if size_average else elbo.sum() # avg = across batch_dim
    return (reconst_loss.mean(), kl.mean())

def _get_probability(q, p, latents, sample_dim, batch_dim, bias):
    log_pz = p.log_joint(sample_dim, batch_dim, latents) # =  p['private'].log_prob.sum(2) + p['shared'].log_prob
    log_q_zCx = q.log_joint(sample_dim, batch_dim, latents) # = q['private'].log_prob.sum(2) + q['shared'].log_prob
    log_joint_qz_marginal, _, log_prod_qzi = q.log_batch_marginal(sample_dim, batch_dim, latents, bias=bias)
    return log_pz, log_joint_qz_marginal, log_prod_qzi, log_q_zCx


#
#
# def _get_prior_params(latent_node):
#     if isinstance(latent_node.dist, RelaxedOneHotCategorical):
#         zeros = torch.zeros(latent_node.value.shape[1:])
#     else:
#         assert isinstance(latent_node.dist, Normal)
#         zeros = torch.zeros(latent_node.value.shape[1:] + (2,))
#     prior_params = Variable(zeros)
#     return prior_params
#
# def _log_importance_weight_matrix(self, batch_size, dataset_size):
#     N = dataset_size
#     M = batch_size - 1
#     strat_weight = (N - M) / (N * M)
#     W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
#     W.view(-1)[::M+1] = 1 / N
#     W.view(-1)[1::M+1] = strat_weight
#     W[M-1, 0] = strat_weight
#     return W.log()
#
#
