from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.context import attach, detach, get_current
import time
import threading

def task(i, parent_context):
    # Attach the parent context in the new thread
    token = attach(parent_context)
    time.sleep(i/2)
    try:
        with tracer.start_as_current_span(f"phoneSpan-{i}") as child_span:
            child_span.set_attribute("phone_number", f"{i}")
            for j in range(3):
                with tracer.start_as_current_span("visqolSpan") as qual_span:
                    time.sleep(i/2)
                    qual_span.set_attribute("voice_quality", 5)
            time.sleep(i)
    finally:
        # Detach the context after the span ends
        detach(token)

trace.set_tracer_provider(
    TracerProvider(
        resource=Resource.create({SERVICE_NAME: "otel-conference"})
    )
)
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)
tracer = trace.get_tracer(__name__)

threads = []
with tracer.start_as_current_span("conferenceSpan"):
    # Capture the current context
    parent_context = get_current()
    for i in range(5):
        t = threading.Thread(target=task, args=(i, parent_context))
        t.start()
        threads.append(t)
        print(f"{i}: Called into conf")
    for t in threads:
        t.join()
print("Call ended")
