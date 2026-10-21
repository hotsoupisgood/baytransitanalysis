#!/bin/bash
cd "$HOME/ytpt"
for i in $(seq 1 900); do
  s=$(.venv/bin/python - <<'PY'
import pathlib, sqlite3
con=sqlite3.connect(pathlib.Path('~/.local/share/ytpt/state.sqlite').expanduser())
b=list(con.execute("SELECT status FROM batches WHERE batch_id='b42'"))[0][0]
d=dict(con.execute("SELECT v.status,count(*) FROM batch_videos bv JOIN videos v ON v.video_id=bv.video_id WHERE bv.batch_id='b42' GROUP BY v.status"))
print(b, d)
PY
)
  echo "$(date -u +%H:%M:%S) $s" >> output/logs/poll_b42.log
  case "$s" in running*) ;; *) echo "SETTLED" >> output/logs/poll_b42.log; break;; esac
  sleep 20
done
