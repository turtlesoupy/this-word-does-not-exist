import 'whatwg-fetch'

function wordURL(word, permalink, relative) {
  const base = relative ? "" : "https://www.thisworddoesnotexist.com";
  return `${base}/w/${encodeURIComponent(word)}` + (permalink ? `/${permalink}` : "");
}


function syncTweetURL(url) {
  let tweetEl = document.getElementById("tweet-a");
  if (tweetEl) {
    tweetEl.href = `https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}`;
  }
}
function syncToWord(word, permalink, pushHistory) {
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
  if (word.example) {
    exampleEl.innerHTML = `"${word.example}"`;
  } else {
    exampleEl.innerHTML = "";
  }

  if (word.syllables && word.syllables.length > 1) {
    syllablesEl.innerHTML = word.syllables.join("<span class='syllable-separator'>&middot;</span>");
  } else {
    syllablesEl.innerHTML = "";
  }

  if (pushHistory && permalink) {
    history.pushState(
      {"word": word, "permalink": permalink}, 
      "",
      wordURL(word.word, permalink, true)
    );
    syncTweetURL(wordURL(word.word, permalink, false));
  }
}

function mobileCheck() {
  let check = false;
  (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
  return check;
};

window.onpopstate = function(event) {
  if (event.state) {
    syncToWord(event.state.word, event.state.permalink, false);
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

  writeButton.classList.remove(["disabled"]);

  // Swap main index page for permalink
  if (history && history.state && history.state.word && history.state.permalink) {
    history.replaceState(
      history.state,
      "",
      wordURL(history.state.word.word, history.state.permalink, true)
    );
  }

  let state = {};

  document.addEventListener('click', event => {
    if (event.target == writeButton) {
      event.preventDefault();

      state.cancelled = false;
      errorText = "something went wrong, try again?"
      definitionEl.style.display = "none";
      writeYourOwnEl.style.display = "";
      hintTextValue.innerHTML = defaultHintText;
      wordEntry.value = "";
      wordEntry.disabled = false;
      wordEntry.focus();
    } else if (event.target == cancelButton) {
      if (state.query_controller) {
        state.query_controller.abort();
      }

      event.preventDefault();
      state.cancelled = true;
      definitionEl.style.display = "";
      writeYourOwnEl.style.display = "none";
      wordEntry.blur();
    }
  }, false);

  let doSubmit = () => {
    wordEntry.disabled = true;
    let word = wordEntry.value;
    hintText.style.display = "none";
    hintTextValue.innerHTML = defaultHintText;
    loadingText.style.display = "";
    document.activeElement.blur()

    recaptchaCallback = (token) => {
      grecaptcha.reset();
      const controller = new AbortController();
      const signal = controller.signal;

      state.query_controller = controller;

      setTimeout(() => controller.abort(), 30000);
      window.fetch(`/define_word?word=${encodeURIComponent(word)}&token=${token}`, {signal})
        .then(res => res.json())
        .then(json => {
          if (!json.word) {
            errorText = "we couldn't define that word"
            throw new Error("couldn't define word");
          }

          syncToWord(json.word, json.permalink, true);
          definitionEl.style.display = "";
          writeYourOwnEl.style.display = "none";
          hintText.style.display = "";
          loadingText.style.display = "none";
        })
        .catch((error) => {
          console.log(error);
          hintTextValue.innerHTML = errorText;
          hintText.style.display = "";
          loadingText.style.display = "none";
          wordEntry.disabled = false;
          wordEntry.focus();
        });
    };

    grecaptcha.execute();
  };
  
  if (mobileCheck()) {
    wordEntry.addEventListener("focusout", e => {
      if (wordEntry.value == "") {
        e.preventDefault();
        return;
      }

      setTimeout(() => {
        if (state.cancelled) {
          return;
        }
        if ((e.isTrusted === true && e.isPrimary === undefined) || e.isPrimary === true || (e.screenX && e.screenX != 0 && e.screenY && e.screenY != 0)) {
          doSubmit();
        }
      }, 250);

    }, false); 
  }

  wordEntryForm.addEventListener("submit", e => {
    e.preventDefault();
    state.cancelled = true;
    doSubmit();
  }, false);
};
