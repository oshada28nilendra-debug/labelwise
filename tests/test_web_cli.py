import io,json,subprocess,sys
from labelwise.web import create_app,SAMPLE

def test_demo_queue_and_benchmark():
    client=create_app().test_client();headers={"X-LabelWise":"local"}
    assert client.get("/").status_code==200
    assert client.get("/",headers={"Host":"evil.test"}).status_code==403
    assert client.post("/api/queue").status_code==403
    assert client.post("/api/queue",headers={"X-LabelWise":"local","Origin":"https://evil.test"}).status_code==403
    q=client.post("/api/queue",data={"sample":"1","size":"7"},headers=headers)
    assert q.status_code==200 and len(q.json["queue"])==7
    b=client.post("/api/benchmark",data={"sample":"1"},headers=headers)
    assert b.status_code==200 and b.json["random_repeats"]==20
    assert b.headers["Cache-Control"]=="no-store"

def test_bad_web_inputs():
    c=create_app().test_client();h={"X-LabelWise":"local"}
    assert c.post("/api/queue",data={"sample":"1","size":"x"},headers=h).status_code==400
    assert c.post("/api/queue",data={},headers=h).status_code==400

def test_cli(tmp_path):
    out=tmp_path/"result.json"
    p=subprocess.run([sys.executable,"-m","labelwise.cli","benchmark","--seed",str(SAMPLE/"seed.csv"),"--pool",str(SAMPLE/"pool-labelled.csv"),"--test",str(SAMPLE/"test.csv"),"--budgets","0","5","--repeats","3","--output",str(out)],capture_output=True,text=True)
    assert p.returncode==0 and json.loads(out.read_text())["budgets"]==[0,5]
    p=subprocess.run([sys.executable,"-m","labelwise.cli","queue","--seed","missing.csv","--pool","missing.csv","--output",str(out)],capture_output=True,text=True)
    assert p.returncode==2
