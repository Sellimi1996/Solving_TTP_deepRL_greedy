class KnapsackDataset(Dataset):

    def __init__(self, items=10, num_samples=1e6, capacity, seed=None):
        super(KnapsackDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset = torch.rand((num_samples, 2, items))
        self.dynamic = torch.zeros(num_samples, 1, items)
        self.num_items = items
        self.items = num_samples
        self.capacity = capacity

    def __len__(self):
        return self.items

    def __getitem__(self, idx):
        return (self.dataset[idx], self.dynamic[idx], [])
    
def update_mask(mask, dynamic, chosen_idx):
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask

def update_fn(self, dynamic, chosen_idx):
        
    wheights = self.static[:,0]
    Wmax = slef.capacity
    selected = chosen_idx.ne(0)
    if selected.any():
        selected_idx = selected.nonzeros(0).sequeeze()
        if current_wheight < Wmax:
            current_wheight = torch.sum(wheigts[selected_idx, selected[selected_idx]]).detach()
            solution[selected_idx, selected[selected_idx]]=1
        else:
            solution[selected_idx, selected[selected_idx]]=0
    return torch.tensor(solution.data, device=dynamic.device)

def reward (static, item_indices):
    idx = item_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    sac = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    profit = torch.sum(sac[:,1:])
    return profit.sum(1).detach()