docker compose -f docker-compose-base.yml --profile elasticsearch --profile kibana up es-kibana-setup
docker compose -f docker-compose-base.yml --profile elasticsearch --profile kibana up -d kibana
