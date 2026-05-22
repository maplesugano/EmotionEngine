# Primitive Affective Unit Candidates

- basis: `data/emotion_code/basis_sweep_L22/ica_k064_seed0.pt`
- decomposer: `ica`  layer=22  k=64
- top_k: 10
- strong_alphas: [-3.0, -1.5, 0.0, 1.5, 3.0]

Each candidate is a basis direction `b_j` that scored well on the encode→steer→re-encode test, has low Plutchik / VAD dominance, and produces a coherent qualitative shift under strong α steering. Fill in `proposed_name` after reading the example generations.

## b26 — PrimitiveScore = 1.562

| metric | value |
| --- | --- |
| self_rho | 0.688 |
| sign_correct | 1 |
| self_delta_pos | 0.004 |
| self_delta_neg | -0.031 |
| causal_strength | 0.018 |
| cross_talk | 0.035 |
| specificity | -0.017 |
| label_dominance | 0.250 |
| mi | 0.129 |
| linear_sep_acc | 0.149 |
| vad_explained | 0.002 |

**proposed_name:** uncertainty in engagement

### α = -3.0

- prompt: `Have you got it now ?`
  → I've got a lot of work to do, so I'll be off. She's always been a bit of a handful, but I'll always love her. She's a bit of a handful, but she's always been a bit of a handful. She's a bit of
- prompt: `Well , when will it be convenient for you ?`
  → I'm free to meet anytime, but I'm not sure if I can make it to your place. She said she'd call me when she's free, and I'm to call her if I'm free before she is. She's always busy, and I'm always free
- prompt: `I come from England .`
  → I've always been fascinated by the way she speaks and she's always been fascinated by my accent. She's always asking me to teach her how to say certain words and phrases in a proper English accent. She's always trying to get me to say things in a proper American accent. She's

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I 'd like to see the look on the faces of the people who were making the rules. The rules that were supposed to be in place to protect the people, but were actually just a bunch of bureaucratic nonsense. The rules that were supposed to be the law, but were actually just a bunch of arbitrary decisions made
- prompt: `Well , when will it be convenient for you ?`
  → I 'd like to schedule the meeting for the 15th of the month. The 15th of the month is the 15th of the month. The 15th of the month is the 15th of the month. The 15th of the month is the 15th of the month
- prompt: `I come from England .`
  → I am a 20-year-old student. I am studying the English language and literature at the university. The university is located in the city of the capital of the country. The city is a big city. The city is the center of the country. The city is the center of the country's culture. The city

## b22 — PrimitiveScore = 1.561

| metric | value |
| --- | --- |
| self_rho | 0.812 |
| sign_correct | 1 |
| self_delta_pos | 0.013 |
| self_delta_neg | -0.036 |
| causal_strength | 0.025 |
| cross_talk | 0.037 |
| specificity | -0.012 |
| label_dominance | 0.500 |
| mi | 0.151 |
| linear_sep_acc | 0.148 |
| vad_explained | 0.009 |

**proposed_name:** uncertainty to assurance

### α = -3.0

- prompt: `Have you got it now ?`
  → I am not sure what you are talking about. Is this a joke or something? I am not sure what is going on here. Is this a joke or something? I am not sure what is going on here. Is this a joke or something? I am not sure what is going on here. Is this a
- prompt: `Well , when will it be convenient for you ?`
  → I am available on the 10th and 11th of next month. Is that okay with you? Or would you like to schedule something else? Please let me know and I will make sure to make it work. Is this a good time for you to meet or talk on the 10th and 11
- prompt: `I come from England .`
  → I am a 25 year old male and I am a student at a university in the UK. I am studying a degree in Business and Management. I am a very friendly and outgoing person and I am very interested in meeting new people and making new friends. I am also very interested in sports and music and I am

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I've been trying to get you to see it for weeks. I've been trying to get you to see that I'm not just a friend, but a friend who cares about you, who wants to help you, who wants to be there for you. I've been trying to get you to
- prompt: `Well , when will it be convenient for you ?`
  → I've been trying to get in touch with you for weeks. - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- prompt: `I come from England .`
  → I've been living in Australia for nearly 20 years. I've been working as a teacher for over 20 years, but I've been teaching English language learners for the past 15 years. I've been teaching at a university for the past 5 years. I'm currently working on

## b17 — PrimitiveScore = 1.502

| metric | value |
| --- | --- |
| self_rho | 0.625 |
| sign_correct | 1 |
| self_delta_pos | 0.006 |
| self_delta_neg | -0.036 |
| causal_strength | 0.021 |
| cross_talk | 0.034 |
| specificity | -0.013 |
| label_dominance | 0.250 |
| mi | 0.136 |
| linear_sep_acc | 0.155 |
| vad_explained | 0.001 |

**proposed_name:** uncertainty to confidence

### α = -3.0

- prompt: `Have you got it now ?`
  → I'm not sure if I'm getting it or not, but I think I might be getting it. I'm not sure if I'm getting it or not, but I think I might be getting it. I'm not sure if I'm getting it or not, but I think I
- prompt: `Well , when will it be convenient for you ?`
  → I'm free to go out for a drink or dinner anytime this week, but I thought maybe we could go out for a walk or something if the weather is nice. I'm not sure if you would like that, but I thought I would suggest it. I would like to get to know you better and
- prompt: `I come from England .`
  → I'm a bit of a fan of the English language and I love to use it in my writing. I also love to use it in my conversations with you, but I know that sometimes it can be a bit tricky for you to understand. So, I thought I would make a list of some of the words

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I mean, I've got it. I've got the whole thing. I've got the whole thing. ( I'm not even kidding. ) I was talking to my friend about this thing, and I was like, " I've got it. " And he was like, " What
- prompt: `Well , when will it be convenient for you ?`
  → I've got a pretty busy schedule, but I'm willing to work around your needs. A. The speaker is asking the other person to schedule a meeting. B. The speaker is asking the other person to schedule a meeting. C. The speaker is asking the other person to schedule a meeting. D.
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student. I am currently studying a degree in Business Administration. I am a very outgoing person. I love meeting new people and making friends. I am very sociable and I love being around other people. I am a very hard worker. I am very dedicated

## b32 — PrimitiveScore = 1.438

| metric | value |
| --- | --- |
| self_rho | 0.750 |
| sign_correct | 1 |
| self_delta_pos | 0.011 |
| self_delta_neg | -0.030 |
| causal_strength | 0.021 |
| cross_talk | 0.035 |
| specificity | -0.014 |
| label_dominance | 0.625 |
| mi | 0.159 |
| linear_sep_acc | 0.135 |
| vad_explained | 0.002 |

**proposed_name:** decisive engagement

### α = -3.0

- prompt: `Have you got it now ?`
  → I've got a lot of work to do and I need to get it done so I can go to the beach and relax. I've been working non stop for like 2 weeks and I need a break. I'm not even going to go to the beach and just sit on the sand and do
- prompt: `Well , when will it be convenient for you ?`
  → I've got a few days free and I can come whenever you want, but I'm not going to be in the city for much longer, so it would be better if we can do it before that. and also, can you tell me what you want to do and how long it will take so i
- prompt: `I come from England .`
  → I am a 25 year old and I am a student at the university in the city of Bologna in Italy. I am studying for a degree in international business and management and I am really enjoying it. I have been in Bologna for about 2 years now and I have fallen in love with the city and

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I've got a good idea of what I want to do. I'm going to go to the library and get some books on the subject. I'm going to read them and learn more. I'm going to make a plan. I'm going to take action. I'm going to
- prompt: `Well , when will it be convenient for you ?`
  → I 'll be here for a while. " " Ah, good. " He smiled. " I 'd like to see you again. " " Me too. " " Perhaps we could have dinner? " " Perhaps. " " Good. " " I 'll call you. " " Good. " "
- prompt: `I come from England .`
  → I am a bit of a traditionalist. I love the old ways of doing things. I love the old songs, the old stories, the old customs. I am a bit of a nostalgic. I love the old-fashioned sense of humour. I love the old-fashioned sense of loyalty. I love the old-fashioned sense

## b38 — PrimitiveScore = 1.438

| metric | value |
| --- | --- |
| self_rho | 0.562 |
| sign_correct | 1 |
| self_delta_pos | 0.018 |
| self_delta_neg | -0.023 |
| causal_strength | 0.021 |
| cross_talk | 0.035 |
| specificity | -0.015 |
| label_dominance | 0.250 |
| mi | 0.137 |
| linear_sep_acc | 0.139 |
| vad_explained | 0.003 |

**proposed_name:** tentative engagement

### α = -3.0

- prompt: `Have you got it now ?`
  → I 't but I 't but I 't but I 't but I 't but I 't but I 't but I 't but I 't but I 't but I 't but I 't but I 't but I 't but I 't but I 't but
- prompt: `Well , when will it be convenient for you ?`
  → I'm not going to be in the area for the next two weeks, but I could come back for the weekend if you want. But I'm not going to be in the area for the next two weeks, but I could come back for the but I'm not going to be in the area for
- prompt: `I come from England .`
  → I am a 25 but I am not a but I am a but I am a but I am a but I am a but I am a but I am a but I am a but I am a but I am a but I am a but I am a but I am a but I am a but I

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I've got a few more things to do before we leave for the airport. We've got to pack up the car and make sure we're all set to go. oh, and i just remembered that we need to stop by the bank on the way to the airport to pick up some cash. u
- prompt: `Well , when will it be convenient for you ?`
  → I've got a few days off from work and I've been thinking about going to the beach. 1. Where are they planning to go? 2. How many days off do they have? 3. What are they thinking about doing at the beach? 4. Who are they talking to
- prompt: `I come from England .`
  → I've been living in the States for a few years now. I love the States, they're so big and open and full of energy. I love the way everyone is always on the go, always doing something. They're so optimistic and friendly. I feel like they're always smiling and laughing

## b49 — PrimitiveScore = 1.436

| metric | value |
| --- | --- |
| self_rho | 0.562 |
| sign_correct | 1 |
| self_delta_pos | 0.015 |
| self_delta_neg | -0.024 |
| causal_strength | 0.019 |
| cross_talk | 0.033 |
| specificity | -0.014 |
| label_dominance | 0.250 |
| mi | 0.125 |
| linear_sep_acc | 0.129 |
| vad_explained | 0.005 |

**proposed_name:** uncertainty to eagerness

### α = -3.0

- prompt: `Have you got it now ?`
  → I'm not even sure I'm the right person to be talking to about this. I'm not even sure I'm the right person to be talking to about anything. I'm just a mess. I'm a mess of emotions and thoughts and feelings and it's all just so much
- prompt: `Well , when will it be convenient for you ?`
  → I'm free any time after 5 pm. I'm really looking forward to it. I'm really excited about the possibility of it. I'm really eager to see it. I've been waiting for it for a long time. I'm really looking forward to the experience. I '
- prompt: `I come from England .`
  → I am a fan of the show and I have been watching it for years. I have seen all the episodes and I have the complete collection of the show on DVD. I am a huge fan of the show and I have a lot of the merchandise. I have the theme music on my phone and I have the show

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I've got a few more things to do before I leave, so I'll be back in a bit. Do you want to come with me to the store to pick up some things? We can go to the store and then come back here to pack. Sounds like a plan, right? What do you
- prompt: `Well , when will it be convenient for you ?`
  → I 'll be here at the store until 6 PM tomorrow. If you can come by then, we can talk about the details. What does the speaker want to talk about? A. A job interview B. A job offer C.. A job interview at a store D. A job at
- prompt: `I come from England .`
  → I am a 25-year-old male. I am a student at a university in the United States. I am studying for a Master's degree in Business Administration ( MBA ). I am currently in my second year of the program. I am a part - time student, so I am only in school for a

## b29 — PrimitiveScore = 1.375

| metric | value |
| --- | --- |
| self_rho | 0.688 |
| sign_correct | 1 |
| self_delta_pos | 0.022 |
| self_delta_neg | -0.014 |
| causal_strength | 0.018 |
| cross_talk | 0.036 |
| specificity | -0.017 |
| label_dominance | 0.625 |
| mi | 0.184 |
| linear_sep_acc | 0.164 |
| vad_explained | 0.001 |

**proposed_name:** commitment uncertainty

### α = -3.0

- prompt: `Have you got it now ?`
  → I've got a lot of work to do today, so I 'll be off. # 1st day of # 1st week of # 1st month of # 1st year of # 1st decade of # 1st century of # 1st millennium of # 1
- prompt: `Well , when will it be convenient for you ?`
  → I 'll be in the city tomorrow, so if you want to meet up, we can do it then. @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
- prompt: `I come from England .`
  → I've been in #Iran for 4 days now & I've been arrested by #IRIGov. I've been accused of spreading anti - govt propaganda & I've been told I 'll be released if I delete my tweets. #FreeMe #FreeIranians pic. tweet.

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I'm not sure what I would do without you. I feel like I'm a complete idiot for not figuring this out sooner. I'm just glad that I have you to help me. I really appreciate it. I feel like I'm a complete idiot for not figuring this out sooner. I '
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on the weekends. I'm also free on Monday and Tuesday. I have a meeting on Wednesday and Thursday. I'm free on Friday. I'm also free on Saturday and Sunday. I am free on the following days: Monday, Tuesday, Friday, Saturday, Sunday. I am
- prompt: `I come from England .`
  → I am a 25 year old male. I have a degree in engineering. I am currently working as a software engineer. I am a fan of the following sports: football, rugby, and tennis. I am a fan of the following music: rock, pop, and hip hop. I am a fan of the

## b62 — PrimitiveScore = 1.251

| metric | value |
| --- | --- |
| self_rho | 0.500 |
| sign_correct | 1 |
| self_delta_pos | 0.025 |
| self_delta_neg | -0.017 |
| causal_strength | 0.021 |
| cross_talk | 0.036 |
| specificity | -0.015 |
| label_dominance | 0.500 |
| mi | 0.131 |
| linear_sep_acc | 0.138 |
| vad_explained | 0.000 |

**proposed_name:** uncertainty in decision-making

### α = -3.0

- prompt: `Have you got it now ?`
  → I've got a few more things to do before I can get some sleep. That's a good idea. I'll get some sleep and then we can talk about it. What does the speaker mean by " it "? That's a good question! The speaker is using " it " as a pron
- prompt: `Well , when will it be convenient for you ?`
  → I'm free to go anytime after 2 pm. That's a good time for me. What's the plan for the rest of the day? That's a good question. I was thinking of going to the movies. That's a good idea. What's the movie you'd
- prompt: `I come from England .`
  → I've been in Australia for 3 years now. I'm a bit of a traditionalist, but I'm also a bit of a rebel. I like to think of myself as a bit of a free spirit. I'm a bit of a worrier, but I'm also a bit

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I am not sure I do. I am not sure I want to. I am not sure I am ready. I am not sure I am ready for the truth. I am not sure I am ready for the truth about myself. I am not sure I am ready for the truth about my life. I am not
- prompt: `Well , when will it be convenient for you ?`
  → I am free on the 15th of next month. I am available on the 15th of next month. I am free on the 15th of next month. I am available on the 15th of next month. I am free on the 15th of next month. I am available on
- prompt: `I come from England .`
  → I am a 25-year-old woman who has been living in the United States for about 5 years. I have been working as a nanny for a family in New York City for about 3 years. I am a very outgoing and energetic person who loves to be around children. I am a very responsible and caring

## b42 — PrimitiveScore = 1.247

| metric | value |
| --- | --- |
| self_rho | 0.562 |
| sign_correct | 1 |
| self_delta_pos | 0.024 |
| self_delta_neg | -0.011 |
| causal_strength | 0.017 |
| cross_talk | 0.034 |
| specificity | -0.017 |
| label_dominance | 0.625 |
| mi | 0.160 |
| linear_sep_acc | 0.146 |
| vad_explained | 0.005 |

**proposed_name:** uncertainty expression

### α = -3.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I can. I'm not sure I should. I'm not sure I can. I'm not sure I should
- prompt: `Well , when will it be convenient for you ?`
  → I 'll be here for the next two weeks, so I can meet you at any time during that period. I 'll be available on weekdays, but I 'll be busy on weekends. So, which day would you like to meet? I 'll be at the same place, so I 'll be
- prompt: `I come from England .`
  → I am a student of the English language, and I have a great passion for the language. I have been studying English for many years, and I have a great understanding of the language. I am able to speak, write, and understand English very well. I am also able to teach English to others, and I

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I've been trying to get it to work for a while now. It's just a simple thing, but it's been driving me crazy. It's a bit of a pain, but I've finally got it sorted. It was just a matter of getting the right combination of things to work
- prompt: `Well , when will it be convenient for you ?`
  → I've got a lot of things to do today and I'd rather not have to deal with it right now. It's not that it's a big deal or anything, it's just that it's something that can wait. If it's really important, you can call me later
- prompt: `I come from England .`
  → I've been living in the States for a few months now and it's been a real culture shock. The people here are so friendly and welcoming, but it's also been really hard to adjust to the different way of life. It's been a big change from the UK, where things are a

## b58 — PrimitiveScore = 1.246

| metric | value |
| --- | --- |
| self_rho | 0.438 |
| sign_correct | 1 |
| self_delta_pos | 0.010 |
| self_delta_neg | -0.012 |
| causal_strength | 0.011 |
| cross_talk | 0.035 |
| specificity | -0.024 |
| label_dominance | 0.375 |
| mi | 0.125 |
| linear_sep_acc | 0.135 |
| vad_explained | 0.001 |

**proposed_name:** uncertainty to confidence

### α = -3.0

- prompt: `Have you got it now ?`
  → I'm not sure he's got it, but he's got a good chance of getting it. He's got a good chance of getting it, but he's not sure he's got it. He's got a good chance of getting it, but he's not sure he
- prompt: `Well , when will it be convenient for you ?`
  → I'll be in the area tomorrow, but I'll be in a meeting from 2 to 4 pm. He said he would be in the area tomorrow, but he would be in a meeting from 2 to 4 pm. He said he would be in the area tomorrow, but he would be
- prompt: `I come from England .`
  → I's that right? ( 1 ) 1.1 I am a British citizen. ( 2 ) 1.2 I am a citizen of the United Kingdom. ( 3 ) 1.3 I am a citizen of the United Kingdom of Great Britain and Northern Ireland. (

### α = +0.0

- prompt: `Have you got it now ?`
  → I'm not sure I do. I'm not sure I want to. I'm not sure I can. I'm not sure I should. I'm not sure I will. I'm not sure I can do it without you. I'm not sure I can do it with you
- prompt: `Well , when will it be convenient for you ?`
  → I'm free on Friday, Saturday, and Sunday. I'm not sure if I can make it on Monday, Tuesday, or Wednesday. I'm not sure if I can make it on Thursday either. I'm not sure if I can make it on Friday either. I'm not sure if
- prompt: `I come from England .`
  → I am a 25 year old male. I am a student at the University of London. I am studying for a degree in Computer Science. I am interested in programming and software development. I am also interested in the field of artificial intelligence. I am a bit of a tech enthusiast and enjoy keeping up to date with

### α = +3.0

- prompt: `Have you got it now ?`
  → I mean, I've got it, and I'm not going to let it go. I'm going to make sure that everyone knows about it, especially the people who have been trying to silence me. I'm going to make sure that everyone knows the truth, no matter how hard it is to
- prompt: `Well , when will it be convenient for you ?`
  → I've got a lot of work to do, but I can always make time for a good friend. So, when do you want to meet up? I'm free most days, especially after 5 pm. How do I know when to meet up with a friend who is busy and wants to meet up
- prompt: `I come from England .`
  → I have a degree in English Literature and a Master's in Creative Writing. I have always been passionate about writing and storytelling, and I have been writing for as long as I can remember. I have written for various publications and have even had a few pieces published in anthologies. I love writing about people, places,
