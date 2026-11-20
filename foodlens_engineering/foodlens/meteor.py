"""Portable stdio wrapper around the standard METEOR 1.5 jar (no score reimplementation)."""
from importlib import import_module
from pathlib import Path
import queue
import subprocess
import threading


def meteor_score(references, predictions, timeout=120):
    module = import_module("pycocoevalcap.meteor.meteor")
    folder = Path(module.__file__).parent
    jar = folder / "meteor-1.5.jar"
    if not jar.is_file():
        raise RuntimeError("pycocoevalcap METEOR jar missing")
    process = subprocess.Popen(["java", "-Xmx2G", "-jar", str(jar), "-", "-", "-stdio", "-l", "en", "-norm"],
        cwd=folder, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8")
    responses = queue.Queue()
    def reader():
        for line in process.stdout:
            responses.put(line.strip())
        responses.put(None)
    threading.Thread(target=reader, daemon=True).start()
    def receive():
        try:
            line = responses.get(timeout=timeout)
        except queue.Empty as error:
            raise RuntimeError("METEOR Java response timed out") from error
        if line is None:
            raise RuntimeError("METEOR Java exited; check Java and bundled paraphrase resources")
        return line
    def send(line):
        process.stdin.write(line + "\n")
        process.stdin.flush()
    try:
        statistics = []
        for sample_id, refs in references.items():
            # Protocol delimiter must not be allowed inside captions.
            clean = lambda text: text.replace("|||", " ").replace("\n", " ")
            send("SCORE ||| " + " ||| ".join([clean(x) for x in refs] + [clean(predictions[sample_id][0])]))
            statistics.append(receive())
        send("EVAL ||| " + " ||| ".join(statistics))
        individual = [float(receive()) for _ in references]
        return float(receive()), individual
    finally:
        process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()
