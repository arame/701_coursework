from word_processing import Word_Processing1

class Sentences1:

    @staticmethod
    def filter(cL):
        sentences1 = []
        for c in cL:
            words = c[0].split()
            words = map(lambda x: Word_Processing1.clean_word(x), words) # Remove "stop" words that do not influence sentiment
            words = list(filter(lambda x:True if len(x) > 0 else False, words))
            sentences1.append(words)

        return sentences1