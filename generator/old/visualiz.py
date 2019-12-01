# visualization
image = torch.unsqueeze(test_dataset[110][0], dim=0)

baselines = [('all zeros', torch.zeros_like(image)),
             ('all 0.5', 0.5*torch.ones_like(image)),             
             ('adv example', pgd_untargeted(model, image, label=8, k=40, eps=0.10, eps_step=0.05, clip_min=0, clip_max=1.0))]

# display baslines
f, axarr = plt.subplots(1,len(baselines), figsize=(18, 16))
for i, b in enumerate(baselines):
    logits = model(b[1])
    c = int(logits.argmax())
    probs = F.softmax(logits)
    axarr[i].imshow(b[1][0, 0, :, :], vmin=0, vmax=1, cmap='gray')
    axarr[i].set_title("{}\n Class {} ({}%)".format(b[0], c, probs[0, c]*100) )
for ax in axarr.flatten():
    ax.set_axis_off()
    
for ax in axarr.flatten():
    ax.set_axis_off()
plt.show()
