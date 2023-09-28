from tensorflow.keras.layers import TextVectorization


if __name__ == '__main__':
    text_vectorization = TextVectorization(output_mode='int')
    dataset = [
        "I write, erase, rewrite",
        "Erase again, and then",
        "A poppy blooms.",
    ]
    text_vectorization.adapt(dataset)
    vocabulary = text_vectorization.get_vocabulary()
    print(vocabulary)
    test_sentence = "I write, rewrite, and still rewrite again"
    encoded_sentence = text_vectorization(test_sentence)
    print(encoded_sentence)
    inverse_vocab = dict(enumerate(vocabulary))

