import json

# female_id = [str(i).zfill(4) for i in range(1, 1683)]
# female_id_test = [str(i).zfill(4) for i in range(1683, 1683+421)]
# male_id = [str(i).zfill(4) for i in range(1, 1361)]
# male_id_test = [str(i).zfill(4) for i in range(1361, 1361+340)]

# female_id = [str(i).zfill(4) for i in range(1, 4001)]
# female_id_test = [str(i).zfill(4) for i in range(4001, 4001+1000)]
# male_id = [str(i).zfill(4) for i in range(1, 4001)]
# male_id_test = [str(i).zfill(4) for i in range(4001, 4001+1000)]

#k-fold
for k in range(1, 5001, 1000):
    female_id = [str(i).zfill(4) for i in range(1, k)] + [str(i).zfill(4) for i in range(k+1000, 5001)]
    female_id_valid = [str(i).zfill(4) for i in range(k, k+1000)]
    female_id_test = [str(i).zfill(4) for i in range(5001, 6001)]
    male_id = [str(i).zfill(4) for i in range(1, k)] + [str(i).zfill(4) for i in range(k+1000, 5001)]
    male_id_valid = [str(i).zfill(4) for i in range(k, k+1000)]
    male_id_test = [str(i).zfill(4) for i in range(5001, 6001)]

    json_dict = {"male": {"train": male_id, "validation": male_id_valid, "test": male_id_test}, "female": {"train": female_id, "validation": female_id_valid, "test": female_id_test}}

    with open("./data10/train_test_data_fold" + str(k) + ".json", "w") as f:
        json.dump(json_dict, f)

