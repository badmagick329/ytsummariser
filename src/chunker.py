from nltk.tokenize import sent_tokenize, word_tokenize
import time
from enum import Enum
from copy import copy


def chunk_text(
    source_text: str,
    max_words_per_chunk: int = 3000,
    overlap: int = 0,
    language="english",
):
    if max_words_per_chunk < 2:
        raise ValueError("max_words_per_chunk must be at least 2 words long.")

    sentences = sent_tokenize(source_text, language=language)
    if not sentences:
        print("Nothing to chunk")
        return []

    chunks = []
    chunk = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        words_in_sent = word_tokenize(sent)
        print(f"-{sent}-\n{len(words_in_sent)}\n")
        if len(words_in_sent) > max_words_per_chunk:
            raise ValueError(
                f"max_words_per_chunk is too small for the sentence: {sent}"
            )

        word_tokens_so_far = word_tokenize(" ".join(chunk))
        print(f"{word_tokens_so_far=}")
        if len(word_tokens_so_far) > max_words_per_chunk:
            chunks.append(chunk)
            chunk = []
            if overlap > 0 and chunks:
                last_chunk = copy(chunks[-1])
                print(f"{last_chunk=} {type(last_chunk)=}")
                chunk = [last_chunk[-(overlap)]]

        if chunk:
            chunk.append(sent)
        else:
            chunk = [sent]

        i += 1
    if chunk:
        chunks.append(chunk)

    return chunks


text = "We got a legendary sequel equivalent to Top Gun Maverick in Machine Learning a few days ago and the Machine Learning scholars sure are eating good these past few weeks. While the first one, Ken was kind of bold in aiming to replace MLP, this time however, we have a good ol' trusty tool in the shed getting upgraded from LSTM to a new and mighty XLSTM sure for extended long short term memory. It is also written by the same author, Sep who has also been shilling XLSTM ever since February which you can find some clues here and there on various sources mentioning how he claimed that XLSTM is better than Mamba with absolutely nothing to back it up during the time when Mamba was rising to fame. But now as the XLSTM paper has been published, we can finally see what who saw in his XLSTM experiments and know if he's high on co-PM or not. What's interesting about XLSTM is that it technically proposed two different LSTM architectures, one's called Scholar LSTM and the other is called Matrix LSTM. Then they can be optionally joined together to create an XLSTM block. But why two architectures? Well, due to the nature of LSTMs, the problems cannot be simply solved with just one architecture and some sort of trade off has to be picked. Something that's really unique to RNN or more specifically, LSTM is that each hidden block relies on the output of a previous hidden block. This is called memory mixing and it is incredibly inefficient due to the need for sequential processing. So while transformers can quickly get the entire output with matrix multiplication, LSTM would have to calculate the output of the individual hidden states sequentially which makes LSTM slow to a point where it is pretty much unscalable. So to overcome this, MLSTM abandoned memory mixing and instead picked up something called Matrix Memory which made LSTMs parallelizable. By also sprinkling exponential gating on top, this solved the old LSTM problem where it will struggle to revisit a stored value when a more similar vector is found. With how MLSTM is designed, its residual block can then be structured similarly to state space models. On the other hand, abandoning memory mixing might be a waste as it is also LSTM's key strength that made it stand out in the first place. Even though it does fail miserably sometimes when trying to revisit values, LSTM was still peak until Transformer was introduced. It was pretty much the best architecture in reinforcement learning back in the days like the Alpha Star model for Starcraft 2 and OpenAI's Dota 2 AI that beat the best professional teams back in 2019 are all built upon LSTM. It excels perfectly at learning abstractions like semantic information."

chunks = chunk_text(text, 100, 1)
print("-----")
print("RESULT")
print("-----")
for chunk in chunks:
    print(chunk)
    print("-----")
    words = word_tokenize(" ".join(chunk))
    print(f"{len(words)}")
    print("-----")
