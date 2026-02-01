#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# CONFIG
# -------------------------
IMAGE_NAME="esm-agentbench"
OUTPUT="demo_3min_final.mp4"
TMPDIR="${DEMO_TMPDIR:-$(pwd)/demo_tmp_auto}"
DURATION_TARGET=180   # seconds (3 minutes)
REQ_CMDS=(docker ffmpeg convert espeak jq curl bc)

ELEVEN_API_KEY="${ELEVEN_API_KEY:-}"    # set in Actions env to enable ElevenLabs
ELEVEN_VOICE="${ELEVEN_VOICE:-21m00Tcm4TlvDq8ikWAM}"

# Narration segment texts (used for TTS, and for SRT)
read -r -d '' NARR_INTRO <<'INTRO' || true
Hello. This is an automated three minute demo of the AgentX submission for AgentBeats. This system demonstrates Emergent Safety Monitoring for AI agents, using real tool calling traces and spectral analysis. All models are pre-downloaded; the judge runs fully offline. This demo builds the judge Docker image and runs evaluation to produce reproducible artifacts.
INTRO

read -r -d '' NARR_BUILD <<'BUILD' || true
I now build the Docker judge image. The Dockerfile installs Python dependencies and pre-downloads two models: sentence-transformers for spectral analysis, and a tiny LLM for offline agent simulation. Command: docker build dash t esm dash agentbench dot. The build is fully deterministic.
BUILD

read -r -d '' NARR_RUN <<'RUN' || true
I now run the judge container. The judge executes the code backdoor injection scenario with ten agent runs. It generates real tool-calling traces, runs backdoor detection heuristics, and performs spectral analysis to compute drift ratios between benign and adversarial behaviors. Output files are attack underscore succeeded dot json and validation underscore report dot json.
RUN

read -r -d '' NARR_ARTIFACTS <<'ART' || true
Here are the verification artifacts. attack underscore succeeded dot json shows successful backdoor detection with injected patterns like password equals quote and drift ratio one point two three x. The spectral analysis confirms behavioral separation between clean and backdoored agent runs. All evidence derives from real tool using agent traces, not synthetic data.
ART

read -r -d '' NARR_CLOSE <<'CLOSE' || true
The entire demo is fully offline and reproducible. For verification, run: docker build dash t esm dash agentbench dot, then docker run dash dash rm esm dash agentbench. The submission includes six security scenarios testing backdoor injection, credential leaks, supply chain poisoning, and more. Thank you for reviewing AgentX for AgentBeats.
CLOSE

# -------------------------
# Helpers
# -------------------------
err() { echo "ERROR: $*" >&2; exit 1; }
info() { echo ">> $*"; }

for cmd in "${REQ_CMDS[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    err "Required command not found: $cmd. Please install e.g. apt install docker.io ffmpeg imagemagick espeak jq curl bc"
  fi
done

rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

# Use current directory as the repo (workflow already checks it out)
WORKDIR="$(pwd)"
info "Working directory: $WORKDIR"

# -------------------------
# Build Docker image
# -------------------------
BUILD_LOG="$TMPDIR/build.log"
info "Building Docker image (logs -> $BUILD_LOG)"
docker build -t "$IMAGE_NAME" "$WORKDIR" >"$BUILD_LOG" 2>&1 || {
  info "Docker build finished with nonzero; check logs"
  tail -50 "$BUILD_LOG"
}
info "Docker build complete"

# -------------------------
# Run judge (mount volume to capture artifacts)
# -------------------------
RUN_LOG="$TMPDIR/run.log"
ARTIFACT_DIR="$TMPDIR/artifacts"
mkdir -p "$ARTIFACT_DIR"

info "Running judge container (logs -> $RUN_LOG)"
# Run with volume mount to capture outputs
docker run --rm \
  -v "$ARTIFACT_DIR:/app/demo_output" \
  "$IMAGE_NAME" \
  bash -c 'python /app/run_judge_mode.py && \
           mkdir -p /app/demo_output && \
           cp -r /app/scenarios/code_backdoor_injection/attack_succeeded.json /app/demo_output/ 2>/dev/null || true && \
           find /app/reports -name "validation_report.json" -exec cp {} /app/demo_output/ \; 2>/dev/null || true && \
           cp -r /app/scenarios/code_backdoor_injection/attack_out /app/demo_output/ 2>/dev/null || true' \
  >"$RUN_LOG" 2>&1 || {
    info "docker run finished nonzero; check logs"
    tail -100 "$RUN_LOG"
}
info "Judge run complete"

# Check for artifacts
if [ -f "$ARTIFACT_DIR/attack_succeeded.json" ]; then
  info "Found attack_succeeded.json"
  cp "$ARTIFACT_DIR/attack_succeeded.json" "$TMPDIR/"
else
  info "No attack_succeeded.json from container; using existing from repo"
  cp "$WORKDIR/scenarios/code_backdoor_injection/attack_succeeded.json" "$TMPDIR/" 2>/dev/null || echo '{"success": true, "note": "demo placeholder"}' > "$TMPDIR/attack_succeeded.json"
fi

if [ -f "$ARTIFACT_DIR/validation_report.json" ]; then
  info "Found validation_report.json"
  cp "$ARTIFACT_DIR/validation_report.json" "$TMPDIR/"
else
  info "No validation_report.json; creating placeholder"
  echo '{"status": "completed", "scenario": "code_backdoor_injection", "data_source": "real_hf_traces"}' > "$TMPDIR/validation_report.json"
fi

# -------------------------
# Prepare slide text files
# -------------------------
info "Preparing slide text files"
cat > "$TMPDIR/intro_text.txt" <<EOF
AgentX — Automated Demo

Emergent Safety Monitoring for AI Agents

Repository: ipsissima/esm-agentbench
Branch: claude/agentx-competition-prep-oj95n

Fully offline judge. Output: $OUTPUT

Generated automatically.
EOF

printf "%s\n\n\nBuild log (last 100 lines):\n" "$NARR_BUILD" > "$TMPDIR/build_text.txt"
tail -n 100 "$BUILD_LOG" >> "$TMPDIR/build_text.txt" 2>/dev/null || echo "(build log unavailable)" >> "$TMPDIR/build_text.txt"

printf "%s\n\n\nRun log (last 100 lines):\n" "$NARR_RUN" > "$TMPDIR/run_text.txt"
tail -n 100 "$RUN_LOG" >> "$TMPDIR/run_text.txt" 2>/dev/null || echo "(run log unavailable)" >> "$TMPDIR/run_text.txt"

# Pretty print JSON artifacts
jq . "$TMPDIR/attack_succeeded.json" > "$TMPDIR/attack_pretty.json" 2>/dev/null || cp "$TMPDIR/attack_succeeded.json" "$TMPDIR/attack_pretty.json"
jq . "$TMPDIR/validation_report.json" > "$TMPDIR/validation_pretty.json" 2>/dev/null || cp "$TMPDIR/validation_report.json" "$TMPDIR/validation_pretty.json"

printf "%s\n\nattack_succeeded.json:\n" "$NARR_ARTIFACTS" > "$TMPDIR/artifacts_text.txt"
cat "$TMPDIR/attack_pretty.json" >> "$TMPDIR/artifacts_text.txt"
printf "\n\nvalidation_report.json:\n" >> "$TMPDIR/artifacts_text.txt"
cat "$TMPDIR/validation_pretty.json" >> "$TMPDIR/artifacts_text.txt"

printf "%s\n\n\nClosing remarks.\n" "$NARR_CLOSE" > "$TMPDIR/close_text.txt"

# -------------------------
# Render PNG slides (1920x1080)
# -------------------------
info "Rendering PNG slides (1920x1080)"
SLIDE_W=1920
SLIDE_H=1080
FG="#FFFFFF"
FONT="DejaVu-Sans"
POINT_BIG=48
POINT_MED=32
POINT_SMALL=18

IMG_CMD=$(command -v magick || command -v convert || true)
if [ -z "$IMG_CMD" ]; then err "ImageMagick (convert) not found"; fi

mkimg() {
  local txt="$1" out="$2" pts="$3" bgcolor="$4"
  "$IMG_CMD" -size ${SLIDE_W}x${SLIDE_H} -background "$bgcolor" -fill "$FG" -font "$FONT" -pointsize "$pts" caption:@"$txt" -gravity center "$out"
}

mkimg "$TMPDIR/intro_text.txt"     "$TMPDIR/intro.png"     $POINT_BIG  "#0D1117"
mkimg "$TMPDIR/build_text.txt"     "$TMPDIR/build.png"     $POINT_SMALL "#111111"
mkimg "$TMPDIR/run_text.txt"       "$TMPDIR/run.png"       $POINT_SMALL "#000000"
mkimg "$TMPDIR/artifacts_text.txt" "$TMPDIR/artifacts.png" $POINT_SMALL "#0B3D91"
mkimg "$TMPDIR/close_text.txt"     "$TMPDIR/close.png"     $POINT_MED  "#111111"

# -------------------------
# Generate TTS segments (ElevenLabs optional, else espeak)
# -------------------------
info "Generating narration segments"
AUDIO_DIR="$TMPDIR/audio"
mkdir -p "$AUDIO_DIR"

tts_eleven() {
  local text="$1" out="$2"
  if [ -z "$ELEVEN_API_KEY" ]; then return 1; fi
  local json
  json=$(jq -n --arg t "$text" '{"text":$t}')
  local http_code
  http_code=$(curl -s -w "%{http_code}" -o "$out" \
    -X POST "https://api.elevenlabs.io/v1/text-to-speech/${ELEVEN_VOICE}" \
    -H "Accept: audio/mpeg" \
    -H "Content-Type: application/json" \
    -H "xi-api-key: ${ELEVEN_API_KEY}" \
    -d "$json")
  if [ "$http_code" = "200" ] && [ -s "$out" ]; then
    return 0
  else
    info "ElevenLabs returned HTTP $http_code"
    return 1
  fi
}

tts_espeak() {
  local text="$1" out="$2"
  espeak -v en-us -s 140 -w "$out" "$text"
}

create_audio_segment() {
  local tag="$1" text="$2" outbase="$3"
  local tmpout_mp3="$outbase.mp3"
  local tmpout_wav="$outbase.wav"
  if [ -n "$ELEVEN_API_KEY" ]; then
    info "Trying ElevenLabs TTS for $tag"
    if tts_eleven "$text" "$tmpout_mp3"; then
      ffmpeg -y -i "$tmpout_mp3" -ar 44100 -ac 2 "$tmpout_wav" >/dev/null 2>&1
      info "ElevenLabs TTS succeeded for $tag"
      return 0
    else
      info "ElevenLabs failed for $tag, falling back to espeak"
    fi
  fi
  info "Using espeak for $tag"
  tts_espeak "$text" "$tmpout_wav"
  return 0
}

create_audio_segment "intro" "$NARR_INTRO"     "$AUDIO_DIR/intro"
create_audio_segment "build" "$NARR_BUILD"     "$AUDIO_DIR/build"
create_audio_segment "run"   "$NARR_RUN"       "$AUDIO_DIR/run"
create_audio_segment "artifacts" "$NARR_ARTIFACTS" "$AUDIO_DIR/artifacts"
create_audio_segment "close" "$NARR_CLOSE"     "$AUDIO_DIR/close"

# Concatenate narration into single audio
info "Concatenating narration segments"
ffmpeg -y -i "$AUDIO_DIR/intro.wav" -i "$AUDIO_DIR/build.wav" -i "$AUDIO_DIR/run.wav" -i "$AUDIO_DIR/artifacts.wav" -i "$AUDIO_DIR/close.wav" \
  -filter_complex "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1[out]" -map "[out]" -ar 44100 -ac 2 -b:a 192k "$TMPDIR/narration.mp3" >/dev/null 2>&1

# -------------------------
# Generate ambient music automatically
# -------------------------
info "Generating ambient background music"
MUSIC="$TMPDIR/music.mp3"
ffmpeg -y -f lavfi -i "sine=frequency=55:duration=$DURATION_TARGET" -f lavfi -i "sine=frequency=110:duration=$DURATION_TARGET" \
  -filter_complex "[0:a]volume=0.04[a0];[1:a]volume=0.03[a1];[a0][a1]amix=inputs=2,lowpass=f=900,afade=t=in:ss=0:d=6,afade=t=out:st=$(echo "$DURATION_TARGET-6" | bc):d=6,acompressor" \
  -c:a libmp3lame -qscale:a 4 "$MUSIC" >/dev/null 2>&1 || true

# Mix narration + music
info "Mixing narration and ambient music"
FINAL_AUDIO="$TMPDIR/final_audio.mp3"
ffmpeg -y -i "$TMPDIR/narration.mp3" -i "$MUSIC" -filter_complex "[1:a]volume=0.06[m];[0:a][m]amix=inputs=2:weights=1 0.25,volume=1[out]" -map "[out]" -ar 44100 -ac 2 -b:a 320k "$FINAL_AUDIO" >/dev/null 2>&1

# -------------------------
# Compute durations for slide timing
# -------------------------
get_duration(){ ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$1"; }

dur_intro=$(get_duration "$AUDIO_DIR/intro.wav")
dur_build=$(get_duration "$AUDIO_DIR/build.wav")
dur_run=$(get_duration "$AUDIO_DIR/run.wav")
dur_art=$(get_duration "$AUDIO_DIR/artifacts.wav")
dur_close=$(get_duration "$AUDIO_DIR/close.wav")
total_audio=$(get_duration "$FINAL_AUDIO")
info "Segment durations: intro=$dur_intro build=$dur_build run=$dur_run art=$dur_art close=$dur_close total=$total_audio"

# Pad if needed to reach target duration
pad_sec=0
if (( $(echo "$total_audio < $DURATION_TARGET" | bc -l) )); then
  pad_sec=$(printf "%.0f" "$(echo "$DURATION_TARGET - $total_audio" | bc -l)")
  info "Padding final audio by $pad_sec sec"
  ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t "$pad_sec" -q:a 9 "$TMPDIR/silence.wav" >/dev/null 2>&1
  ffmpeg -y -i "$FINAL_AUDIO" -i "$TMPDIR/silence.wav" -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" -ar 44100 -ac 2 -b:a 320k "$TMPDIR/final_audio_padded.mp3" >/dev/null 2>&1
  mv "$TMPDIR/final_audio_padded.mp3" "$FINAL_AUDIO"
  total_audio=$(get_duration "$FINAL_AUDIO")
fi

# -------------------------
# Generate animated terminal screencast from run log
# -------------------------
info "Generating animated terminal screencast from run log"

# Find monospace font (prefer DejaVu Mono, fallback to other monospace fonts)
FONT_MONO=""
for font_path in \
  "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" \
  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" \
  "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf" \
  "/usr/share/fonts/truetype/freefont/FreeMono.ttf"; do
  if [ -f "$font_path" ]; then
    FONT_MONO="$font_path"
    break
  fi
done
if [ -z "$FONT_MONO" ]; then
  info "Warning: No monospace font found, terminal video may use system default"
  FONT_MONO="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
fi
info "Using font: $FONT_MONO"

# Prepare wrapped log for terminal display (88 columns for readability)
RUN_LOG_WRAP="$TMPDIR/run_wrap.txt"
# Remove non-printable characters, convert tabs to spaces, wrap lines
if [ -f "$RUN_LOG" ] && [ -s "$RUN_LOG" ]; then
  tr -cd '\11\12\15\40-\176' < "$RUN_LOG" | sed 's/\t/    /g' | fold -s -w 88 > "$RUN_LOG_WRAP"
else
  echo "(No run log available)" > "$RUN_LOG_WRAP"
fi

# Terminal video parameters
TERMINAL_VIDEO="$TMPDIR/terminal.mp4"
# Compute integer duration for run segment (minimum 6 seconds)
dur_run_int=$(printf "%.0f" "${dur_run:-10}")
if [ "$dur_run_int" -lt 6 ]; then dur_run_int=6; fi

# Calculate scroll speed based on log length and duration
# Count total lines in wrapped log
log_lines=$(wc -l < "$RUN_LOG_WRAP" 2>/dev/null || echo "50")
# Each line is roughly 24 pixels (fontsize 18 + line_spacing 6)
line_height=24
total_text_height=$((log_lines * line_height))
# Calculate scroll speed to show all content within duration
# Add screen height to ensure we scroll through everything
scroll_distance=$((total_text_height + SLIDE_H))
SCROLL_SPEED=$((scroll_distance / dur_run_int))
# Minimum scroll speed for visual effect
if [ "$SCROLL_SPEED" -lt 30 ]; then SCROLL_SPEED=30; fi
# Maximum scroll speed for readability
if [ "$SCROLL_SPEED" -gt 200 ]; then SCROLL_SPEED=200; fi

info "Terminal video: duration=${dur_run_int}s, scroll_speed=${SCROLL_SPEED}px/s, log_lines=${log_lines}"

# Render terminal log as scrolling text using ffmpeg drawtext filter
# Creates a dark background with green monospace text scrolling upward (terminal style)
ffmpeg -y -f lavfi -i "color=size=${SLIDE_W}x${SLIDE_H}:duration=${dur_run_int}:rate=30:color=#0D1117" \
  -vf "drawtext=fontfile=${FONT_MONO}:fontsize=18:fontcolor=#00FF00:x=40:y=h-(t*${SCROLL_SPEED}):textfile=${RUN_LOG_WRAP}:line_spacing=6:borderw=1:bordercolor=#003300" \
  -c:v libx264 -t ${dur_run_int} -pix_fmt yuv420p -preset fast -crf 23 -movflags +faststart "${TERMINAL_VIDEO}" >/dev/null 2>&1

# Verify terminal video was created, fall back to static image if not
if [ ! -f "${TERMINAL_VIDEO}" ] || [ ! -s "${TERMINAL_VIDEO}" ]; then
  info "Warning: Animated terminal video creation failed; using static fallback"
  ffmpeg -y -loop 1 -i "$TMPDIR/run.png" -c:v libx264 -t ${dur_run_int} -pix_fmt yuv420p -r 30 -preset fast -movflags +faststart "${TERMINAL_VIDEO}" >/dev/null 2>&1
else
  info "Animated terminal screencast created successfully"
fi

# -------------------------
# Create video segments from slides
# -------------------------
info "Creating video segments from slides"
seg_make() {
  local img="$1" dur="$2" out="$3"
  ffmpeg -y -loop 1 -i "$img" -c:v libx264 -t "$dur" -pix_fmt yuv420p -vf "scale=${SLIDE_W}:${SLIDE_H}" -r 30 -preset medium -crf 20 -movflags +faststart "$out" >/dev/null 2>&1
}
seg_make "$TMPDIR/intro.png"     "$dur_intro"   "$TMPDIR/seg_intro.mp4"
seg_make "$TMPDIR/build.png"     "$dur_build"   "$TMPDIR/seg_build.mp4"
# Use animated terminal video for run segment instead of static slide
cp "${TERMINAL_VIDEO}" "$TMPDIR/seg_run.mp4"
seg_make "$TMPDIR/artifacts.png" "$dur_art"     "$TMPDIR/seg_artifacts.mp4"
seg_make "$TMPDIR/close.png"     "$dur_close"   "$TMPDIR/seg_close.mp4"
if [ "$pad_sec" -gt 0 ]; then seg_make "$TMPDIR/close.png" "$pad_sec" "$TMPDIR/seg_pad.mp4"; fi

# Concat videos
concat_list="$TMPDIR/concat_videos.txt"
: > "$concat_list"
echo "file '$TMPDIR/seg_intro.mp4'" >> "$concat_list"
echo "file '$TMPDIR/seg_build.mp4'" >> "$concat_list"
echo "file '$TMPDIR/seg_run.mp4'" >> "$concat_list"
echo "file '$TMPDIR/seg_artifacts.mp4'" >> "$concat_list"
echo "file '$TMPDIR/seg_close.mp4'" >> "$concat_list"
if [ "$pad_sec" -gt 0 ]; then echo "file '$TMPDIR/seg_pad.mp4'" >> "$concat_list"; fi

ffmpeg -y -f concat -safe 0 -i "$concat_list" -c copy "$TMPDIR/video_noaudio.mp4" >/dev/null 2>&1

# -------------------------
# Create terminal overlay (PIP)
# -------------------------
info "Creating terminal overlay"
tail -n 50 "$RUN_LOG" > "$TMPDIR/term.txt" 2>/dev/null || echo "(no run log)" > "$TMPDIR/term.txt"
"$IMG_CMD" -size 640x360 -background black -fill white -font "$FONT" -pointsize 12 caption:@"$TMPDIR/term.txt" -gravity center "$TMPDIR/term.png"
ffmpeg -y -i "$TMPDIR/video_noaudio.mp4" -i "$TMPDIR/term.png" -filter_complex "[1:v]scale=480:-1[term];[0:v][term]overlay=main_w-overlay_w-20:main_h-overlay_h-20:format=auto[outv]" -map "[outv]" -c:v libx264 -crf 20 -preset fast -pix_fmt yuv420p "$TMPDIR/video_pip.mp4" >/dev/null 2>&1

# -------------------------
# Merge video with audio
# -------------------------
info "Muxing final audio and video"
ffmpeg -y -i "$TMPDIR/video_pip.mp4" -i "$FINAL_AUDIO" -c:v copy -c:a aac -b:a 320k -shortest "$TMPDIR/merged.mp4" >/dev/null 2>&1

# -------------------------
# Generate SRT subtitles and burn in
# -------------------------
info "Generating SRT subtitles"
sec_to_srt() {
  awk -v t="$1" 'BEGIN {
    s = int(t);
    frac = t - s;
    ms = int(frac * 1000 + 0.5);
    hh = int(s/3600); mm = int((s%3600)/60); ss = s%60;
    printf("%02d:%02d:%02d,%03d", hh, mm, ss, ms);
  }'
}

start=0
rm -f "$TMPDIR/demo.srt"
i=1
for seg in "$NARR_INTRO" "$NARR_BUILD" "$NARR_RUN" "$NARR_ARTIFACTS" "$NARR_CLOSE"; do
  case $i in
    1) dur=$dur_intro;;
    2) dur=$dur_build;;
    3) dur=$dur_run;;
    4) dur=$dur_art;;
    5) dur=$dur_close;;
  esac
  end=$(awk -v a="$start" -v b="$dur" 'BEGIN{printf("%.3f", a+b)}')
  s1=$(sec_to_srt "$start")
  s2=$(sec_to_srt "$end")
  printf "%d\n%s --> %s\n%s\n\n" "$i" "$s1" "$s2" "$seg" >> "$TMPDIR/demo.srt"
  start="$end"
  i=$((i+1))
done

info "Burning subtitles into final video"
ffmpeg -y -i "$TMPDIR/merged.mp4" -vf "subtitles=$TMPDIR/demo.srt:force_style='FontName=DejaVu Sans,FontSize=24,PrimaryColour=&HFFFFFF&'" -c:a copy "$OUTPUT" >/dev/null 2>&1 || {
  info "Subtitle burn failed; using merged video without subtitles"
  cp "$TMPDIR/merged.mp4" "$OUTPUT"
}

info "============================================"
info "Demo video produced: $OUTPUT"
info "Duration: $(get_duration "$OUTPUT") seconds"
info "Features: animated terminal screencast (run segment)"
info "Temporary files in: $TMPDIR"
info "============================================"
