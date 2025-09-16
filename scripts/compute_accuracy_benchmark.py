import pandas as pd

# Load the benchmark dataset
dataset_tsv = "/mloscratch/users/nemo/benchmarking/GMAI-MMBench/GMAI-MMBench_VAL_new.tsv"
dataset = pd.read_csv(dataset_tsv, sep="\t", header=0)

# Extract the answers
answers_gd = dataset["answer"].tolist()
nb_possible_answers = dataset.apply(lambda x: 4 if x["E"] != x["E"] else 5, axis=1).tolist()

# Extract the answers of MultiMeditron
with open("/mloscratch/users/lmartins/openmeditron/MultiMeditron/scripts/outputs_benchmarks/answers_benchmark_GMAI-MMBench_1306.txt") as f:
    answers_mm = list(f.read())

N = len(answers_mm)
nb_correct = sum(a.lower() == b.lower() for a, b in zip(answers_gd, answers_mm))
N_answered = sum(a != "?" for a in answers_mm)

print(N, "answers by MultiMeditron")
print(nb_correct, "correct answers, the accuracy is", nb_correct / N)
print("Precision", nb_correct / N_answered)
print("The probably to guess correctly based on a uniform random draw is", sum(1 / nb for nb, _ in zip(nb_possible_answers, answers_mm)) / len(answers_mm))

#for batch in range(0, len(answers_mm), 80):
#    print("".join(answers_mm[batch:batch+80]))
#    print("".join(answers_gd[batch:batch+80]))
#    print("-------------")
