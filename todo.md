- [x] make sure train code leverages GPU `11/21`
- [x] verify output of model is correct  `11/22`
- [ ] split training into multiple phases
  - [x] cell that trains and save model params `11/25` 
  - [ ] <del> cell that loads model checkpoint and outputs a video </del>
  - [ ] <del> cell that loads model checkpoint and outputs still images </del>
  - [ ] cell that loads model checkpoint and outputs still images + video  
    - [x] MVP `11/25`
    - [x] Verify output is reasonable `11/26`
    - [ ] remove unecessary code

- [ ] design choice - gigantic args object may be confusing for inexperienced users
  - [x] decide what args should be kept constant, and which should be changed by the user `11/26`
    - [x] ie rather than a million boolean vars get rid of them and just turn if statements into  functions `11/26`
    - [x] add new args `11/26`
      - [x] ex num of output images `11/26`
  - [ ] make sure code functions the same as before

- [ ] Google colab is not as generous as it was years ago
  - [x] frequent disconnects; implement being able to continue training `11/26`
