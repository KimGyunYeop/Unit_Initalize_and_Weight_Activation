import wandb

api = wandb.Api()

runs = api.runs(path="isnlp_lab/unit_init_glue")

a = set()
for r in runs:
    # if r.state == "crashed" or r.state == "failed":
    #     continue
    
    if "noact_" in r.name:
        if "no_act" in r.name or "elu" in r.name:
            print(r.name)
            print(str(r.name).replace("noact_",""))
            
            r.name = str(r.name).replace("noact_","")
            r.update()
    