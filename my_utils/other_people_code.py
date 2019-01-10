

size = 512
overlap = 128
x_split = 5
y_split = 5
for i in range(x_split):
    for j in range(y_split):
        patch = image[
            i*size - i*overlap: (i+1)*size - i*overlap,
            j*size - j*overlap: (j+1)*size - j*overlap
        ]
        dst = os.path.join(dstdir, f'{name}_x{i}_y{j}.png')
        cv2.imwrite(dst, patch)



for i,(images,target) in enumerate(train_loader):

    # 1. input output
    images = images.cuda(non_blocking=True)
    target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
    outputs = model(images)
    loss = criterion(outputs,target)

    # 2.1 loss regularization
    loss = loss/accumulation_steps

    # 2.2 back propagation
    loss.backward()

    # 3. update parameters of net
    if(i%accumulation_steps)==0:
        optimizer.step()        # update parameters of net
        optimizer.zero_grad()   # reset gradient



