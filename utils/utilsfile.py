from pandas import read_csv


def calculate_weights_class(labels):
    """Computes class weight determinations for the following attributes: valence, arousal, 
    and live streaming weights, based on a Pandas data frame of labels.
    """
    annotations = read_csv(labels)
    total_labels = len(annotations.index)

    number_valence_negative = annotations['Valence_Negative'].sum()
    number_valence_neutral = annotations['Valence_Neutral'].sum()
    number_valence_positive = annotations['Valence_Positive'].sum()
    number_arousal_negative = annotations[' Arousal_Low'].sum()
    number_arousal_neutral = annotations[' Arousal_Neutral'].sum()
    number_arousal_postive = annotations[' Arousal_High'].sum()
    routing_number = annotations[' Routing'].sum()
    procurement_number = annotations[' Procurement'].sum()
    respawning_number = annotations[' Respawning'].sum()
    exploring_number = annotations[' Exploring'].sum()
    fighting_number = annotations[' Fighting'].sum()
    punching_number = annotations[' Punching'].sum()
    defending_number = annotations[' Defending'].sum()
    defeated_number = annotations[' Defeated'].sum()

    valence_weight_negative = float(total_labels)/(3 * number_valence_negative)
    valence_weight_neutral = float(total_labels)/(3 * number_valence_neutral)
    valence_weight_positive = float(total_labels)/(3 * number_valence_positive)

    tot = valence_weight_negative + valence_weight_neutral + valence_weight_positive

    valence_weight_negative = valence_weight_negative/tot
    valence_weight_neutral = valence_weight_neutral/tot
    valence_weight_positive = valence_weight_positive/tot

    arousal_weight_negative = float(total_labels) / (3 * number_arousal_negative)
    arousal_weight_neutral = float(total_labels) / (3 * number_arousal_neutral)
    arousal_weight_positive = float(total_labels) / (3 * number_arousal_postive)

    tot = arousal_weight_negative + arousal_weight_neutral + arousal_weight_positive

    arousal_weight_negative = arousal_weight_negative / tot
    arousal_weight_neutral = arousal_weight_neutral / tot
    arousal_weight_positive = arousal_weight_positive / tot

    routing_weight = float(total_labels) / (8 * routing_number)
    procurement_weight = float(total_labels) / (8 * procurement_number)
    respawning_weight = float(total_labels) / (8 * respawning_number)
    exploring_weight = float(total_labels) / (8 * exploring_number)
    fighting_weight = float(total_labels) / (8 * fighting_number)
    punching_weight = float(total_labels) / (8 * punching_number)
    defending_weight = float(total_labels) / (8 * defending_number)
    defeated_weight = float(total_labels) / (8 * defeated_number)

    tot = routing_weight + procurement_weight + respawning_weight + exploring_weight + fighting_weight + punching_weight + defending_weight + defeated_weight

    routing_weight = routing_weight/tot
    procurement_weight = procurement_weight/tot
    respawning_weight = respawning_weight/tot
    exploring_weight = exploring_weight/tot
    fighting_weight = fighting_weight/tot
    punching_weight = punching_weight/tot
    defending_weight = defending_weight/tot
    defeated_weight = defeated_weight/tot

    weights_valence = {0: valence_weight_negative, 1: valence_weight_neutral, 2: valence_weight_positive}
    weights_arousal = {0: arousal_weight_negative, 1: arousal_weight_neutral, 2: arousal_weight_positive}
    weights_live_streaming = {
        0: routing_weight, 1: procurement_weight, 2: respawning_weight, 3: exploring_weight,
        4: fighting_weight, 5: punching_weight, 6: defending_weight, 7: defeated_weight
    }
    return weights_valence, weights_arousal, weights_live_streaming

def get_confusion_matrix(model, generate_data, batch_size, videos_number, flag_model, outfile):
    """Generates a file that stores the confusion matrix for a specified model.

    Parameters:

    model: The trained model.
    generate_data: A data source for computing the confusion matrix.
    batch_size: The batch size used for the data generation process.
    videos_number: The total number of videos in the data source.
    flag_model: Specifies the model type ("both," "game," or "emo").
    outfile: The name of the file where the confusion matrix will be saved.
    """
    if flag_model == "both":
        get_confusion_matrix_both(model, generate_data, batch_size, videos_number, outfile)
    elif flag_model == "game":
        get_confusion_matrix_game(model, generate_data, batch_size, videos_number, outfile)
    else:
        get_confusion_matrix_emo(model, generate_data, batch_size, videos_number, outfile)


def get_confusion_matrix_both(model, generate_data, batch_size, videos_number, outfile):
    """Generates a file that stores the confusion matrix for a model trained to work with both live streaming and emotional data.

    Parameters:

    model: The trained model.
    generate_data: Data to generate the confusion matrix for.
    batch_size: The batch size for processing the generate_data.
    videos_number: The number of videos in the generate_data.
    outfile: The name of the file where the confusion matrix will be saved.
    """
    confusion_matrix_valence = [[0 for _ in range(3)] for _ in range(3)]
    confusion_matrix_arousal = [[0 for _ in range(3)] for _ in range(3)]
    confusion_matrix_live_streaming = [[0 for _ in range(8)] for _ in range(8)]

    output_file = open(outfile, "w")

    for _ in range(int(videos_number / batch_size)):
        data = next(generate_data)
        res = model.predict(data[0], batch_size=batch_size)
        for i in range(batch_size):
            confusion_matrix_valence[res[0][i].argmax()][data[1][0][i].argmax()] += 1
            confusion_matrix_arousal[res[0][i].argmax()][data[1][0][i].argmax()] += 1
            confusion_matrix_live_streaming[res[2][i].argmax()][data[1][2][i].argmax()] += 1

    output_file.write("Val Conf Mat" + "\n")
    print("Val Conf Mat")
    for i in range(3):
        output_file.write(str(confusion_matrix_valence[i]) + "\n")
        print(confusion_matrix_valence[i])

    output_file.write("Aro Conf Mat" + "\n")
    print("Aro Conf Mat")
    for i in range(3):
        output_file.write(str(confusion_matrix_arousal[i]) + "\n")
        print(confusion_matrix_arousal[i])

    output_file.write("live_streaming Conf Mat" + "\n")
    print("live_streaming Conf Mat")
    for i in range(8):
        output_file.write(str(confusion_matrix_live_streaming[i]) + "\n")
        print(confusion_matrix_live_streaming[i])

    output_file.close()


def get_confusion_matrix_game(model, generate_data, batch_size, videos_number, outfile):
    """Generates a file that stores the confusion matrix for a specified model trained for a game.

    Parameters:

    model: The trained model.
    generate_data: The data used to calculate the confusion matrix.
    batch_size: The batch size for processing the generate_data.
    videos_number: The total number of videos in the generate_data.
    outfile: The name of the file where the confusion matrix will be saved.
    """
    confusion_matrix_live_streaming = [[0 for _ in range(8)] for _ in range(8)]

    output_file = open(outfile, "w")

    for _ in range(int(videos_number / batch_size)):
        data = next(generate_data)
        res = model.predict(data[0], batch_size=batch_size)
        for i in range(batch_size):
            confusion_matrix_live_streaming[res[i].argmax()][data[1][0][i].argmax()] += 1

    output_file.write("live_streaming Conf Mat" + "\n")
    print("live_streaming Conf Mat")
    for i in range(8):
        output_file.write(str(confusion_matrix_live_streaming[i]) + "\n")
        print(confusion_matrix_live_streaming[i])

    output_file.close()


def get_confusion_matrix_emo(model, generate_data, batch_size, videos_number, outfile):
    """Generate a confusion matrix file for a trained emotion model using the following parameters:

    model: The trained model.
    generate_data: Data used for generating the confusion matrix.
    batch_size: The batch size for processing the generate_data.
    videos_number: The total number of videos in the generate_data.
    outfile: The desired file name to store the confusion matrix.
    """
    confusion_matrix_valence = [[0 for _ in range(3)] for _ in range(3)]
    confusion_matrix_arousal = [[0 for _ in range(3)] for _ in range(3)]

    output_file = open(outfile, "w")

    for _ in range(int(videos_number / batch_size)):
        data = next(generate_data)
        res = model.predict(data[0], batch_size=batch_size)
        for i in range(batch_size):
                confusion_matrix_valence[res[0][i].argmax()][data[1][0][i].argmax()] += 1
                confusion_matrix_arousal[res[0][i].argmax()][data[1][0][i].argmax()] += 1

    output_file.write("Val Conf Mat" + "\n")
    print("Val Conf Mat")
    for i in range(3):
        output_file.write(str(confusion_matrix_valence[i]) + "\n")
        print(confusion_matrix_valence[i])

    output_file.write("Aro Conf Mat" + "\n")
    print("Aro Conf Mat")
    for i in range(3):
        output_file.write(str(confusion_matrix_arousal[i]) + "\n")
        print(confusion_matrix_arousal[i])

    output_file.close()

