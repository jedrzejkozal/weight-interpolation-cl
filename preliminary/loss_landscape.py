from interpolate_networks import *
import torch
import torch.nn as nn
import loss_landscape
import loss_landscapes.metrics
import numpy as np
import matplotlib.pyplot as plt

import utils
import dataset
from train import resnet18


@torch.no_grad()
def main():
    utils.seed_everything(42)
    steps = 10

    loss_function = nn.CrossEntropyLoss()
    train_dataloader_part1, train_dataloader_part2, test_dataloader = dataset.get_dataloaders('cifar100', train_halves=True)

    model1 = resnet18()
    utils.load_model(model1, 'resnet18_v1_part1')
    model1.to('cuda')
    model2 = resnet18()
    utils.load_model(model2, 'resnet18_v1_part2')
    model2.to('cuda')
    model3 = resnet18()
    utils.load_model(model3, 'resnet18_v1_part1')
    model3.to('cuda')
    model3 = permute_nework(model2, model3, train_dataloader_part1, test_dataloader)
    model4 = resnet18()
    model4 = interpolation(model2, model3, model4, train_dataloader_part1)

    coordinates, dir_one, dir_two, start_point = loss_landscape.three_models(model1, model2, model3, model4, normalization='filter', steps=steps)
    coor1, coor2, coor3, coor4 = coordinates

    X, y = get_data(train_dataloader_part1)
    metric = loss_landscapes.metrics.Loss(loss_function, X, y)
    # landscape = loss_landscape.random_plane(model, metric, normalization='filter', steps=steps)
    model = resnet18()
    load_params(model, start_point)
    model_wrapper = loss_landscape.wrap_model(model)
    start_point = model_wrapper.get_module_parameters()
    landscape = loss_landscape.get_grid(metric, steps, model_wrapper, start_point, dir_one, dir_two)

    # landscape = get_landscape()
    print(landscape)
    delta = 1.0
    border = steps // 2
    x = np.arange(-border, border, delta)
    y = np.arange(-border, border, delta)
    X, Y = np.meshgrid(x, y)

    loss_min = np.log10(landscape.min())
    loss_max = np.log10(landscape.max())
    levels = np.logspace(loss_min, loss_max, num=10)
    levels_finegrained = np.linspace(landscape.min(), min(2*landscape.min(), levels[1]), num=5)
    levels = [levels[0]] + list(levels_finegrained[1:-1]) + list(levels[1:])
    print(levels)
    print(levels_finegrained)
    # exit()

    CS = plt.contour(X, Y, landscape,)  # levels=levels)
    plt.clabel(CS, inline=True, fontsize=10, fmt='%1.1e')
    plt.plot(*coor1, 'rx', label='model 1')
    plt.plot(*coor2, 'bx', label='model 2')
    plt.plot(*coor3, 'gx', label='premuted model 1')
    plt.plot(*coor4, 'mo', label='interpolation')
    plt.legend()
    plt.show()


def load_params(model, parameters):
    params_iter = iter(parameters)
    for name, _ in model.named_parameters():
        setattr(model, name, next(params_iter))


def permute_nework(source_network, premutation_nework, train_loader, test_loader):
    source_network = add_junctures(source_network)
    premutation_nework = add_junctures(premutation_nework)
    premutation_nework = permute_network(train_loader, test_loader, source_network, premutation_nework)
    premutation_nework = remove_junctures(premutation_nework)
    source_network = remove_junctures(source_network)
    return premutation_nework


def interpolation(sournce_network, premutation_nework, output_network, train_loader, alpha=0.5):
    mix_weights(output_network, alpha, sournce_network, premutation_nework)
    reset_bn_stats(output_network, train_loader)
    return output_network


def get_data(train_dataloader):
    X = []
    y = []
    for batch_X, batch_y in train_dataloader:
        X.append(batch_X)
        y.append(batch_y)
        # break
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    X = X.to('cuda')
    y = y.to('cuda')
    return X, y


def get_landscape():
    landscape = np.array([[7.69712543e+00, 7.70526981e+00, 7.71408367e+00, 7.72336817e+00,
                           7.72480202e+00, 7.71942377e+00, 7.70576143e+00, 7.68322372e+00,
                           7.65175629e+00, 7.61323023e+00, 7.56614876e+00, 7.51192808e+00,
                           7.45147562e+00, 7.38661432e+00, 7.31948566e+00, 7.25185871e+00,
                           7.18588352e+00, 7.12284040e+00, 7.06255865e+00, 7.00432634e+00,],
                          [7.62281799e+00, 7.62272358e+00, 7.62656021e+00, 7.63017511e+00,
                           7.63175249e+00, 7.63129759e+00, 7.62160110e+00, 7.60087824e+00,
                           7.56964827e+00, 7.53043699e+00, 7.48246241e+00, 7.42608738e+00,
                           7.36341333e+00, 7.29617834e+00, 7.22716761e+00, 7.15850306e+00,
                           7.09213448e+00, 7.02833271e+00, 6.96640682e+00, 6.90618944e+00,],
                          [7.53651667e+00, 7.53296232e+00, 7.53162813e+00, 7.52673435e+00,
                           7.51556492e+00, 7.49767065e+00, 7.46910810e+00, 7.42947769e+00,
                           7.38167953e+00, 7.32561922e+00, 7.26296186e+00, 7.19574451e+00,
                           7.12707853e+00, 7.05811071e+00, 6.99089050e+00, 6.92583895e+00,
                           6.86281013e+00, 6.80075312e+00, 6.73963737e+00, 6.67975664e+00,],
                          [7.45826006e+00, 7.44374752e+00, 7.43313646e+00, 7.42282200e+00,
                           7.40820932e+00, 7.38669586e+00, 7.35712242e+00, 7.31832886e+00,
                           7.27098751e+00, 7.21602583e+00, 7.15339851e+00, 7.08712196e+00,
                           7.01896477e+00, 6.94998932e+00, 6.88181019e+00, 6.81540394e+00,
                           6.75074863e+00, 6.68714428e+00, 6.62595749e+00, 6.56953001e+00,],
                          [7.34819794e+00, 7.32898045e+00, 7.30727005e+00, 7.28168440e+00,
                           7.24928236e+00, 7.20792437e+00, 7.15880632e+00, 7.10288668e+00,
                           7.04055119e+00, 6.97445917e+00, 6.90585613e+00, 6.83633709e+00,
                           6.76655865e+00, 6.69840574e+00, 6.63156509e+00, 6.56835604e+00,
                           6.51116180e+00, 6.46807003e+00, 6.45817184e+00, 6.53271437e+00,],
                          [7.27567434e+00, 7.24722290e+00, 7.21779299e+00, 7.18596458e+00,
                           7.14922380e+00, 7.10471296e+00, 7.05252075e+00, 6.99424505e+00,
                           6.93008852e+00, 6.86181927e+00, 6.79102802e+00, 6.71924877e+00,
                           6.64793730e+00, 6.57785511e+00, 6.51038790e+00, 6.45033741e+00,
                           6.40742922e+00, 6.40148830e+00, 6.47643423e+00, 6.69785929e+00,],
                          [7.13841486e+00, 7.09954548e+00, 7.05686665e+00, 7.00787878e+00,
                           6.95230865e+00, 6.89054918e+00, 6.82324553e+00, 6.75088739e+00,
                           6.67561817e+00, 6.59985638e+00, 6.52787495e+00, 6.46037245e+00,
                           6.39818001e+00, 6.35090446e+00, 6.35650015e+00, 6.49319887e+00,
                           6.78683996e+00, 7.13242197e+00, 7.68507004e+00, 8.39946175e+00,],
                          [7.06554508e+00, 7.01891565e+00, 6.96982193e+00, 6.91581917e+00,
                           6.85590553e+00, 6.78923893e+00, 6.71703005e+00, 6.63987207e+00,
                           6.55972815e+00, 6.47927332e+00, 6.40511370e+00, 6.34267330e+00,
                           6.29555941e+00, 6.31262445e+00, 6.52806044e+00, 6.89697742e+00,
                           7.15521479e+00, 7.37874174e+00, 7.82190132e+00, 8.65298271e+00,],
                          [6.88193989e+00, 6.82186270e+00, 6.75657558e+00, 6.68525505e+00,
                           6.60890198e+00, 6.52868176e+00, 6.44543934e+00, 6.36173058e+00,
                           6.29258680e+00, 6.25228548e+00, 6.28497744e+00, 6.52238798e+00,
                           6.82390022e+00, 6.86246395e+00, 6.82411957e+00, 7.10224771e+00,
                           8.85288429e+00, 1.59104204e+01, 3.88969269e+01, 1.32437363e+02,],
                          [6.78522921e+00, 6.72065210e+00, 6.65257120e+00, 6.58130121e+00,
                           6.50758314e+00, 6.43362522e+00, 6.35757113e+00, 6.29614925e+00,
                           6.28747749e+00, 6.34015226e+00, 6.46562767e+00, 6.49047565e+00,
                           6.36046505e+00, 6.49259758e+00, 7.29688311e+00, 1.20445576e+01,
                           2.55935059e+01, 7.30447464e+01, 2.68374542e+02, 1.02302448e+03,],
                          [6.55067301e+00, 6.49023104e+00, 6.43816710e+00, 6.39659452e+00,
                           6.38662767e+00, 6.45066118e+00, 6.51067352e+00, 6.41242933e+00,
                           6.27595329e+00, 6.51197624e+00, 6.78503609e+00, 1.03660717e+01,
                           1.99083557e+01, 5.13155403e+01, 1.68524063e+02, 6.21867371e+02,
                           2.41431885e+03, 9.56774707e+03, 3.84215312e+04, 1.59352500e+05,],
                          [6.49171829e+00, 6.47916603e+00, 6.50183105e+00, 6.58477736e+00,
                           6.73111200e+00, 6.75204802e+00, 6.47110987e+00, 6.36466455e+00,
                           6.62076044e+00, 9.37855721e+00, 1.70836449e+01, 4.13140411e+01,
                           1.23497559e+02, 4.21081329e+02, 1.54766223e+03, 6.12956885e+03,
                           2.50111660e+04, 1.04201266e+05, 4.37776469e+05, 1.82923275e+06,],
                          [6.96607065e+00, 7.15747452e+00, 7.08014679e+00, 6.69216299e+00,
                           6.82426167e+00, 8.90113735e+00, 1.57140284e+01, 3.76936531e+01,
                           1.04375732e+02, 3.26549622e+02, 1.13678638e+03, 4.23521289e+03,
                           1.68849043e+04, 7.07761953e+04, 2.98188500e+05, 1.24092612e+06,
                           5.11873550e+06, 2.07148060e+07, 8.16933680e+07, 3.05126336e+08,],
                          [7.76581955e+00, 7.61622477e+00, 7.65336752e+00, 9.18204784e+00,
                           1.58640184e+01, 3.73554077e+01, 1.00463814e+02, 2.90772888e+02,
                           9.27760803e+02, 3.30236182e+03, 1.23824375e+04, 4.99180195e+04,
                           2.11256266e+05, 8.90418312e+05, 3.74336675e+06, 1.53227890e+07,
                           6.11843880e+07, 2.36839232e+08, 8.79787136e+08, 3.08374938e+09,],
                          [1.84566917e+01, 4.16857681e+01, 1.05980598e+02, 2.92619019e+02,
                           8.53777100e+02, 2.78560522e+03, 1.00681641e+04, 3.96880391e+04,
                           1.68822688e+05, 7.28133312e+05, 3.08395125e+06, 1.27822400e+07,
                           5.16220160e+07, 2.00255664e+08, 7.43133760e+08, 2.61034701e+09,
                           8.68483686e+09, 2.78143795e+10, 8.46650655e+10, 2.44281786e+11,],
                          [1.20313370e+02, 3.07748291e+02, 8.71613525e+02, 2.67662085e+03,
                           9.16005078e+03, 3.41014102e+04, 1.38099234e+05, 5.90912562e+05,
                           2.52412200e+06, 1.05851980e+07, 4.33302200e+07, 1.70587968e+08,
                           6.41508544e+08, 2.27395814e+09, 7.64258611e+09, 2.44656292e+10,
                           7.44554086e+10, 2.17516081e+11, 6.02169606e+11, 1.60105090e+12,],
                          [9.09297168e+03, 3.18294375e+04, 1.20586258e+05, 4.91502312e+05,
                           2.07101188e+06, 8.78906500e+06, 3.64613760e+07, 1.46368512e+08,
                           5.56630848e+08, 2.00149619e+09, 6.80306534e+09, 2.19552010e+10,
                           6.72361595e+10, 1.95589865e+11, 5.45927922e+11, 1.45413715e+12,
                           3.73071125e+12, 9.25970517e+12, 2.22014430e+13, 5.22987388e+13,],
                          [1.13999578e+05, 4.32289438e+05, 1.76006250e+06, 7.34868850e+06,
                           3.07173400e+07, 1.24928872e+08, 4.84424256e+08, 1.76629555e+09,
                           6.11092582e+09, 1.99906509e+10, 6.19777720e+10, 1.80694090e+11,
                           5.00205126e+11, 1.33289725e+12, 3.43755299e+12, 8.53338174e+12,
                           2.05886786e+13, 4.80570098e+13, 1.09270620e+14, 2.47366628e+14,],
                          [2.58874780e+07, 1.05913160e+08, 4.16209568e+08, 1.55826918e+09,
                           5.45493043e+09, 1.81593395e+10, 5.68246600e+10, 1.68622916e+11,
                           4.69348450e+11, 1.24382754e+12, 3.18806321e+12, 7.91915214e+12,
                           1.91184975e+13, 4.49597806e+13, 1.02533662e+14, 2.28129067e+14,
                           5.03383522e+14, 1.10366781e+15, 2.37220761e+15, 4.99622808e+15,],
                          [3.55104128e+08, 1.35138406e+09, 4.85793741e+09, 1.63466547e+10,
                           5.20542618e+10, 1.55804647e+11, 4.39712514e+11, 1.17481577e+12,
                           3.00657594e+12, 7.45321949e+12, 1.79568325e+13, 4.22631828e+13,
                           9.66000137e+13, 2.15297349e+14, 4.71386049e+14, 1.02150120e+15,
                           2.20712732e+15, 4.66989056e+15, 9.71179508e+15, 1.98643483e+16,]])
    return landscape


if __name__ == '__main__':
    main()
