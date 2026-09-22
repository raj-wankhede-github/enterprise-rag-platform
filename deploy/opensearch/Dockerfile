FROM opensearchproject/opensearch:2.19.0

# The stock image bundles ~24 plugins. Each costs heap in classloaders, thread pools and
# scheduled jobs, and several run background work on every node forever. We use exactly one of
# them, so the rest come out.
#
# Two things about `opensearch-plugin remove` that are easy to get wrong:
#   * it accepts ONE plugin per invocation -- passing a list makes it print its help text and
#     exit non-zero, which a trailing `|| true` will happily swallow, leaving every plugin
#     installed and the image looking like it worked;
#   * plugins have dependencies, so dependents must be removed before what they depend on.
#     The order below is deliberate.
#
# Kept on purpose: opensearch-knn (the entire point), opensearch-job-scheduler (dependency of
# several things and cheap), opensearch-ltr and query-insights (learning-to-rank and slow-query
# analysis are both on the roadmap), plus the security and codec plugins.
RUN set -eu; \
    for plugin in \
        opensearch-skills \
        opensearch-neural-search \
        opensearch-security-analytics \
        opensearch-alerting \
        opensearch-anomaly-detection \
        opensearch-index-management \
        opensearch-reports-scheduler \
        opensearch-observability \
        opensearch-notifications \
        opensearch-notifications-core \
        opensearch-ml \
        opensearch-flow-framework \
        opensearch-cross-cluster-replication \
        opensearch-asynchronous-search \
        opensearch-geospatial \
        opensearch-sql \
        opensearch-performance-analyzer \
    ; do \
        echo "removing ${plugin}"; \
        /usr/share/opensearch/bin/opensearch-plugin remove "${plugin}"; \
    done

# Fail the build rather than ship an image that cannot do vector search. Without this the
# removal loop could take out a dependency of knn and nobody would find out until query time.
RUN /usr/share/opensearch/bin/opensearch-plugin list | grep -qx opensearch-knn \
    || { echo "FATAL: opensearch-knn was removed; this image is useless for vector search"; exit 1; }
