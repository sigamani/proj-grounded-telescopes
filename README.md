├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── /src               # optional: put batch_infer.py / preprocessing (not written but will be pydantic serialisation of pii) / utilities code and stuff
├── /monitoring        # configs for Grafana / Prometheus / Loki
│    ├── prometheus.yml
│    ├── dashboards/
│    └── ...
├── /ray-tmp           # host-volume for Ray’s spill/session directories (dunno if this is a great place due to size constraints maybe /tmp?)
└── .dockerignore      (yet to be birthed)
