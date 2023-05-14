import torch

@torch.no_grad()
def main():
    torch.manual_seed(42)

    cumulative_mean: torch.TensorType = None
    M2: torch.TensorType = None
    n:int = 0

    all_samples = []

    for _ in range(100):
        tensor = torch.randn(size=(32, 3, 224, 224))
        all_samples.append(tensor)

        # whole_dataset = torch.cat(all_samples, dim=0)
        # mean = torch.mean(whole_dataset, dim=0)
        # std = torch.std(whole_dataset, dim=0)
        # print(f'sample: mean = {mean[:, 0, 0]}, std = {std[:, 0, 0]}')

        if cumulative_mean == None:
            cumulative_mean = torch.zeros(size=tensor.shape[1:])

        if M2 == None:
            M2 = torch.zeros(size=tensor.shape[1:])

        for new_img in tensor:
            n += 1
            delta = new_img - cumulative_mean
            cumulative_mean += delta / n
            delta2 = new_img - cumulative_mean
            M2 += delta * delta2

            # print(f'cumulative mean = {mean_cumulative[:, 0, 0]}, std = {std_cumulative[:, 0, 0]}')
            # print()


    whole_dataset = torch.cat(all_samples, dim=0)
    mean = torch.mean(whole_dataset, dim=0)
    std = torch.std(whole_dataset, dim=0)
    print(f'sample: mean = {mean[:, 0, 0]}, std = {std[:, 0, 0]}')

    cumulative_var = M2 / (n-1)
    std_cumulative = torch.sqrt(cumulative_var)

    print(f'cumulative mean = {cumulative_mean[:, 0, 0]}, std = {std_cumulative[:, 0, 0]}')



if __name__ == '__main__':
    main()