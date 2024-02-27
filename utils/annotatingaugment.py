from pandas import read_csv


def get_l_rep_class(annotations):
	"""Obtains the classes that appear the least and the most frequently across all labels.

	Input: A pandas data frame containing annotations.

	Output: A tuple consisting of the class that is most commonly represented and the class that is 
	least represented (both in string format).
	"""
	dictionary_valence = {
		'Valence_Negative': annotations['Valence_Negative'].sum(),
		'Valence_Neutral': annotations['Valence_Neutral'].sum(),
		'Valence_Positive': annotations['Valence_Positive'].sum()
	}

	dictionary_arousal = {
        ' Arousal_Low': annotations[' Arousal_Low'].sum(),
		' Arousal_Neutral': annotations[' Arousal_Neutral'].sum(),
		' Arousal_High': annotations[' Arousal_High'].sum()
	}

	dictionary_live_streaming = {
		' Routing': annotations[' Routing'].sum(),
		' Procurement': annotations[' Procurement'].sum(),
		' Respawning': annotations[' Respawning'].sum(),
		' Exploring': annotations[' Exploring'].sum(),
		' Fighting': annotations[' Fighting'].sum(),
		' Punching': annotations[' Punching'].sum(),
		' Defending': annotations[' Defending'].sum(),
		' Defeated': annotations[' Defeated'].sum()
	}

	total = dictionary_valence['Valence_Negative']+dictionary_valence['Valence_Neutral']+dictionary_valence['Valence_Positive']

	valence_smallest = min(dictionary_valence, key=dictionary_valence.get)
	arousal_smallest = min(dictionary_arousal, key=dictionary_arousal.get)
	live_streaming_smallest = min(dictionary_live_streaming, key=dictionary_live_streaming.get)

	weighted_valence_smallest = float(annotations[valence_smallest].sum()/total)/(1./3.)
	weighted_arousal_smallest = float(annotations[arousal_smallest].sum()/total)/(1./3.)
	weighted_live_streaming_smallest = float(annotations[live_streaming_smallest].sum()/total)/(1./8.)

	if(weighted_valence_smallest < weighted_arousal_smallest) and (weighted_valence_smallest < weighted_live_streaming_smallest):
		smallest = valence_smallest
	elif weighted_arousal_smallest < weighted_live_streaming_smallest:
		smallest = arousal_smallest
	else:
		smallest = live_streaming_smallest

	valence_biggest = max(dictionary_valence, key=dictionary_valence.get)
	arousal_biggest = max(dictionary_arousal, key=dictionary_arousal.get)
	live_streaming_biggest = max(dictionary_live_streaming, key=dictionary_live_streaming.get)

	weighted_valence_biggest = float(annotations[valence_biggest].sum()/total)/(1./3.)
	weighted_arousal_biggest = float(annotations[arousal_biggest].sum()/total)/(1./3.)
	weighted_live_streaming_biggest = float(annotations[live_streaming_biggest].sum()/total)/(1./8.)

	if(weighted_valence_biggest > weighted_arousal_biggest) and (weighted_valence_biggest > weighted_live_streaming_biggest):
		biggest = valence_biggest
	elif weighted_arousal_biggest > weighted_live_streaming_biggest:
		biggest = arousal_biggest
	else:
		biggest = live_streaming_biggest

	return smallest, biggest


def main():
	"""Processes a CSV containing annotations and conducts oversampling to address data imbalance.
	"""
	annotations = read_csv("../train.csv")
	limit = 5517

	for i in range(0, limit):
		
		class_l, class_m = get_l_rep_class(annotations)

		data = annotations.loc[(annotations[class_l] == 1) & (annotations[class_m] == 0)]
		if len(data.index) == 0:
			print("Exiting Early at ", i)
			break
		samp = data.sample(1)
		annotations = annotations.append(samp)

	annotations.to_csv("../train_augmented.csv")

if __name__ == "__main__":
	main()
