import json

def open_json(file_path, mode="r"):
    with open(file_path, mode) as f:
        data = json.load(f)
    return data

def save_json(data, save_path, mode="w", sort_keys=True, indent=4):
    with open(save_path, mode) as f:
        json.dump(data, f, sort_keys=sort_keys, indent=indent)

if __name__ == "__main__":
    path = "/media/x/Wlty/nnUNet_workspace/nnUNet_preprocessed/Dataset101_MixedData/splits_final.json"
    lst = open_json(path)
    add_data = ["MixedData_{:04d}".format(i) for i in range(202, 202+120)]
    for dic in lst:
        dic["train"] = dic["train"] + add_data

    save_json(lst, path)