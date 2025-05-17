import torch
import torch.autograd as autograd


def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss

# 现在用的是这个？
def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    # used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    # perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    # target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    # scores = scorenet(perturbed_samples, labels)
    # target = target.view(target.shape[0], -1)
    # scores = scores.view(scores.shape[0], -1)

    print("😈")
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    print("used_sigmas shape:", used_sigmas.shape)  # 例如 [batch_size, 1, 1, 1...]

    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    print("perturbed_samples shape:", perturbed_samples.shape)  # 通常和 samples 形状相同

    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    print("target shape:", target.shape)  # 和 samples 形状相同

    scores = scorenet(perturbed_samples, labels)
    print("scores shape:", scores.shape)  # 根据模型输出，通常是 [batch_size, C, H, W] 或其他

    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    print("target reshaped shape:", target.shape)  # [batch_size, feature_dim]
    print("scores reshaped shape:", scores.shape)  # [batch_size, feature_dim]

    print("👼")
    print("scores shape:", scores.shape)           # [batch_size, feature_dim]
    print("target shape:", target.shape)           # [batch_size, feature_dim]
    print("used_sigmas shape:", used_sigmas.shape) # [batch_size, 1, 1, ...]（和samples的channel/height/width匹配）

    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)

def anneal_dsm_score_estimation_l1(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = torch.abs((scores - target)).sum(dim=-1) * used_sigmas.squeeze()

    return loss.mean(dim=0)


def anneal_dsm_score_estimation_l1l2(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1/2. * torch.abs((scores - target)).sum(dim=-1) * used_sigmas.squeeze() + 1 / 4. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)
