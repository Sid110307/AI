# TODO

- Use move semantics: When returning values from functions, instead of making a copy of the object, consider using move
  semantics to avoid unnecessary copying. For example, in cleanText function, instead of returning text, we can return
  it as std::move(text).
- Reuse resources: In the validateDataset function, we open the file "cornell_movie_dialogs_corpus.zip" twice, once for
  downloading and again for extracting. We can reuse the downloaded file instead of downloading it again. Similarly, we
  can reuse the file stream instead of creating a new one each time.
- Use reserve for vectors and strings: If you know the size of a vector or string beforehand, consider using reserve to
  avoid reallocations.
- Use const reference where possible: In the cleanText function, the text parameter can be passed as a const reference (
  const std::string& text) to avoid unnecessary copying.
- Use emplace for inserting elements: Instead of using insert to insert elements into an unordered map, consider using
  emplace to avoid unnecessary copying.
- Avoid unnecessary copies: In the forward function of the Chatbot class, instead of creating a copy of the output of
  the LSTM module (std::get<0>(lstm->forward(embedded))), we can use auto& to avoid the unnecessary copy.
- Use at instead of [] for unordered map: If you're not sure whether a key exists in an unordered map, consider using at
  instead of [] to avoid creating a new element if the key is not found.
- Additional dataset: https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train_socratic.jsonl

