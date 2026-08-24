---
layout: page
title: Soccer AI Commentary
description: Automatic soccer commentary generation — mixed spatial-temporal attention and vision-language talking-head synthesis
img: /assets/paris-2024-olympics-soccer.jpg
importance: 4
---

Soccer is the world's most-watched sport, yet professional commentary is a skilled, real-time human craft that scales only to the matches someone is paid to cover. The long tail of lower-league, amateur, and archived games goes silent, and visually impaired fans are locked out of the audio description that makes a broadcast followable. Automatic commentary closes that gap: it turns raw footage into scalable match narration, instant highlights, and accessible play-by-play, sitting at the intersection of video understanding, language generation, and speech and video synthesis.

The work is an end-to-end pipeline that converts broadcast soccer footage into narrated, anchor-presented highlights. It runs in two stages. First, a game-understanding stage watches the frames and produces structured textual commentary — a play-by-play grounded in what actually happened on the pitch. Second, a presentation stage voices that commentary and renders it as a synchronized talking-head anchor, so the output reads like a real broadcast segment rather than a caption.

### Game understanding

The understanding stage is built on an action-spotting backbone that uses mixed spatial-temporal attention. Rather than treating a clip as a flat sequence, it jointly models *where* the action is — spatial structure across the pitch — and *when* events unfold — temporal structure across the clip — and from that emits coherent commentary tied to key moments such as goals, fouls, and passes. The approach was published at TENCON 2022 (IEEE Region 10 Conference, Hong Kong), training on the SoccerNet benchmark, and is implemented on a TWINS-style spatial-attention vision backbone adapted to 1-D temporal sequences with PyTorch Lightning.

### From commentary to a talking head

The generated commentary text is then voiced by a text-to-speech engine — either a general engine such as EdgeTTS or a cloned-voice model via GPT-SoVITS or CosyVoice — and mapped onto a talking-head anchor using MuseTalk lip-sync. The result closes the loop from raw footage to a broadcast-ready segment, runs on a single consumer GPU, and is exposed through a web interface for interactive demos. Because the commentary itself is produced by a language model, the same pipeline can shift voice, language, or style without retraining the understanding stage.

## Demo

{% include video.liquid path="/assets/videos/soccer-commentary-full-demo.mp4" controls=true width="100%" caption="Full demo recording — end-to-end commentary + talking-head." %}

<!-- <figure>
  <video src="https://github.com/user-attachments/assets/c24946a7-0f81-490c-840a-9f5c3c9300aa" controls width="100%" preload="metadata"></video>
  <figcaption class="caption">Pure lip-sync demo.</figcaption>
</figure>

<figure>
  <video src="https://github.com/user-attachments/assets/41373536-0d6b-4d08-be6a-fc39612c4176" controls width="100%" preload="metadata"></video>
  <figcaption class="caption">Commentary result — EdgeTTS voice.</figcaption>
</figure> -->

{% include video.liquid path="/assets/videos/soccer-commentary-gpt-sovits.mp4" controls=true width="100%" caption="Commentary result — GPT-SoVITS voice." %}


## Related Repo

| Repo | Description |
|------|-------------|
| [MixSaT](https://github.com/allanchan339/MixSaT) | Mixed Spatial and Temporal Attention for automatic soccer game commentary (TENCON 2022) |
| [VLM_Soccer_Commentator_THG](https://github.com/allanchan339/VLM_Soccer_Commentator_THG) | Cantonese soccer commentary + synchronized talking-head video (TTS + lip-sync) |
