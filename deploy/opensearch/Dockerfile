FROM opensearchproject/opensearch:2.19.0

# The stock image bundles a dozen plugins we do not use. Each one costs heap in classloaders,
# thread pools and scheduled jobs -- together enough to matter even on a production node, and
# decisive on a small one. opensearch-knn stays: it is the entire point of this deployment.
RUN /usr/share/opensearch/bin/opensearch-plugin remove --purge \
      opensearch-alerting \
      opensearch-anomaly-detection \
      opensearch-asynchronous-search \
      opensearch-cross-cluster-replication \
      opensearch-flow-framework \
      opensearch-geospatial \
      opensearch-index-management \
      opensearch-ml \
      opensearch-notifications \
      opensearch-notifications-core \
      opensearch-observability \
      opensearch-performance-analyzer \
      opensearch-reports-scheduler \
      opensearch-security-analytics \
      opensearch-sql \
      || true
