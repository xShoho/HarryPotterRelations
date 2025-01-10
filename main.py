import os
import pandas as pd
import spacy

NER = spacy.load("en_core_web_sm")
NER.max_length = 1500000

books = [b for b in os.scandir('books') if '.txt' in b.name]
characters_df = pd.read_csv('./HarryPotterCharacters.csv')
window_size = 5
relationships = []


def filter_characters(sentence_entity_list, characters_df):
    return [sentence_entity for sentence_entity in sentence_entity_list
            if sentence_entity in list(characters_df.character_name)
            or sentence_entity in list(characters_df.character_surname)
            or sentence_entity in list(characters_df.character_name + ' ' + characters_df.character_surname)]


for book in books:
    book_text = open(book).read()
    book_doc = NER(book_text)

    sentence_entity_df = []

    for sentence in book_doc.sents:
        sentence_entity_list = [sentence_entity.text for sentence_entity in sentence.ents]
        sentence_entity_df.append({"sentence": sentence, "entities": sentence_entity_list})

    sentence_entity_df = pd.DataFrame(sentence_entity_df)

    # Get characters from sentences
    sentence_entity_df['character_entities'] = sentence_entity_df['entities'].apply(lambda x: filter_characters(x, characters_df))

    # Remove sentences that don't have any characters
    sentence_entity_df = sentence_entity_df[sentence_entity_df['character_entities'].map(len) > 0]

    for i in range(sentence_entity_df.index[-1]):
        end_i = min(i + window_size, sentence_entity_df.index[-1])
        characters_list = sum((sentence_entity_df.loc[i: end_i].character_entities), [])

        # Remove duplicated characters that are next to each other
        characters_unique = [characters_list[i] for i in range(len(characters_list))
                            if i == 0 or characters_list[i] != characters_list[i - 1]]
        
        if len(characters_unique) > 1:
            for index, first_character in enumerate(characters_unique[:-1]):
                second_character = characters_unique[index + 1]
                relationships.append({'first_character': first_character, 'second_character': second_character})


relationships_df = pd.DataFrame(relationships)
relationships_df.to_csv('./HarryPotterRelations.csv', index= False)