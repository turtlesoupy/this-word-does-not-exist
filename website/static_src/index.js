import 'whatwg-fetch'

function syncToWord(word) {
  let posEl = document.getElementById("definition-pos");
  let wordEl = document.getElementById("definition-word");
  let syllablesEl = document.getElementById("definition-syllables");
  let definitionEl = document.getElementById("definition-definition");
  let exampleEl = document.getElementById("definition-example");

  posEl.innerHTML = word.pos;
  if (!posEl.innerHTML.endsWith("]")) {
    posEl.innerHTML += ".";
  }

  wordEl.innerHTML = word.word;
  definitionEl.innerHTML = word.definition;
  if (word.examples && word.examples.length > 0) {
    exampleEl.innerHTML = word.examples[0];
  } else {
    exampleEl.innerHTML = "";
  }

  if (word.syllables && word.syllables.length > 1) {
    syllablesEl.innerHTML = word.syllables.join("<span class='syllable-separator'>&middot;</span>");
  } else {
    syllablesEl.innerHTML = "";
  }
}

window.onload = () => {
  let definitionEl = document.getElementById("definition-zone");
  let writeYourOwnEl = document.getElementById("write-your-own");
  let wordEntry = document.getElementById("word-entry");
  let writeButton = document.getElementById("write-button");
  let hintText = document.getElementById("hint-text");
  let loadingText = document.getElementById("word-loading");
  let wordEntryForm = document.getElementById("word-entry-form");
  let cancelButton = document.getElementById("word-entry-cancel")
  let hintTextValue = document.getElementById("hint-text-value");
  let defaultHintText = hintTextValue.innerHTML;
  var errorText = "something went wrong, try again?"

  document.addEventListener('click', event => {
    if (event.target == writeButton) {
      event.preventDefault();

      errorText = "something went wrong, try again?"
      definitionEl.style.display = "none";
      writeYourOwnEl.style.display = "";
      hintTextValue.innerHTML = defaultHintText;
      wordEntry.value = "";
      wordEntry.disabled = false;
      wordEntry.focus();
    } else if (event.target == cancelButton) {
      event.preventDefault();

      definitionEl.style.display = "";
      writeYourOwnEl.style.display = "none";
      wordEntry.blur();
    }
  }, false);


  wordEntryForm.addEventListener("submit", e => {
    e.preventDefault();
    wordEntry.disabled = true;
    let word = wordEntry.value;
    hintText.style.display = "none";
    hintTextValue.innerHTML = defaultHintText;
    loadingText.style.display = "";
    document.activeElement.blur()
    window.fetch(`/define_word?word=${encodeURIComponent(word)}`)
      .then(res => res.json())
      .then(json => {
        console.log(json);
        if (!json.word) {
          errorText = "we couldn't define that word"
          throw new Error("couldn't define word");
        }

        syncToWord(json.word);
        definitionEl.style.display = "";
        writeYourOwnEl.style.display = "none";
        hintText.style.display = "";
        loadingText.style.display = "none";
      })
      .catch((error) => {
        hintTextValue.innerHTML = errorText;
        hintText.style.display = "";
        loadingText.style.display = "none";
        wordEntry.disabled = false;
        wordEntry.focus();
      });
  }, false);
};
