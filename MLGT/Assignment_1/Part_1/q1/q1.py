import json
import numpy as np
import matplotlib.pyplot as plt


def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """populate the 'samples' list from the desired distribution"""

    samples = []

    # TODO: first generate random numbers from the uniform distribution
    if distribution == "cauchy":
        random_numbers = []
        random_numbers = np.random.rand(num_samples)
        i = 0
        for y in random_numbers:
            random_numbers[i] = round(y, 4)
            i = i + 1
        # print(random_numbers)

        for x in random_numbers:
            samples.append(np.tan(np.pi * (x - 0.5)))

    elif distribution == "exponential":
        random_numbers = np.random.rand(num_samples)
        i = 0
        for y in random_numbers:
            random_numbers[i] = round(y, 4)
            i = i + 1
        for z in random_numbers:
            samples.append((-1 / kwargs["lambda"]) * np.log(1 - z))

    # print(samples)

    # END TODO

    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)

        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        histo = "q1" + distribution + ".png"

        plt.hist(samples, bins=100)
        plt.title(distribution)
        plt.xlabel("Samples_Generated")
        plt.ylabel("frequency")
        plt.savefig(histo)
        # plt.show()
        plt.clf()

        # END TODO
