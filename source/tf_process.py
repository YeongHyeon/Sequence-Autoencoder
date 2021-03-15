import os, shutil
import numpy as np
import matplotlib.pyplot as plt

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def post_processing(coord):

    """
    coord_x, coord_y
    0, 0 # start
    1, 0 # pause
    1, 1 # end
    """

    list_total, list_temp = [], []
    for idx, _ in enumerate(coord):
        # if(idx == 0): continue
        coord_x, coord_y = coord[idx, 0], coord[idx, 1]
        # if(coord_x == 1 and (coord_y == 0 or coord_y == 1)):
        if(coord_x == 0 and (coord_y == 0 or coord_y == 1)):
            if(len(list_temp) > 0):
                list_total.append(np.asarray(list_temp))
            list_temp = []
            if(coord_y == 0): continue
            elif(coord_y == 1): break
        list_temp.append([coord_x, coord_y])

    if(len(list_total) == 0):
        if(len(list_temp) > 0):
            list_total.append(np.asarray(list_temp))
    return list_total

def show_pen(x, y, label="", save_png=""):

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.title(label)
    plt.plot(x[:, 0], x[:, 1], alpha=0.1)
    x_r = post_processing(coord=x)
    for idx, _ in enumerate(x_r):
        plt.scatter(x_r[idx][0, 0], x_r[idx][0, 1], marker='x', c='red')
        plt.plot(x_r[idx][:, 0], x_r[idx][:, 1])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.title('Generated')
    plt.plot(y[:, 0], y[:, 1], alpha=0.1)
    y_r = post_processing(coord=y[:-1])
    for idx, _ in enumerate(y_r):
        plt.scatter(y_r[idx][0, 0], y_r[idx][0, 1], marker='x', c='red')
        plt.plot(y_r[idx][:, 0], y_r[idx][:, 1])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_png)
    plt.close()

def training(neuralnet, dataset, epochs, batch_size):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    save_dir = "results_tr"
    make_dir(path=save_dir, refresh=True)

    iteration = 0
    for epoch in range(epochs):

        while(True):
            x_tr, y_tr, terminate = dataset.next_batch(batch_size=1, train=True)
            if(terminate): break
            try:
                step_dict = neuralnet.step(x=x_tr, iteration=iteration, training=True)
            except:
                dataset.reset_idx()
                break
            else:
                if(iteration % 10 == 0):
                    show_pen(x=x_tr[0], y=step_dict['y_hat'][0], label=y_tr[0], \
                        save_png=os.path.join(save_dir, 'epoch_%04d-iteration_%08d.png' %(epoch, iteration)))

                iteration += 1

        print("Epoch [%d / %d] (%d iteration) Loss: %.5f" \
            %(epoch, epochs, iteration, step_dict['loss']))

        neuralnet.save_parameter(model='model_checker', epoch=epoch)

def test(neuralnet, dataset, batch_size):

    print("\nTest...")
    neuralnet.load_parameter(model='model_checker')

    save_dir = "results_te"
    make_dir(path=save_dir, refresh=True)

    num_test = 0
    while(True):
        x_te, y_te, terminate = dataset.next_batch(batch_size=batch_size, train=False)
        try: step_dict = neuralnet.step(x=x_te, training=False)
        except: pass
        else:
            for idx in range(x_te.shape[0]):
                show_pen(x=x_te[0], y=step_dict['y_hat'][idx], label=y_te[idx], \
                    save_png=os.path.join(save_dir, 'test_%08d.png' %(num_test)))
                num_test += 1

        if(terminate): break
