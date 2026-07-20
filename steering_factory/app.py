from __future__ import annotations

import os
import sys

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steering_factory.config import load_config, TargetModelConfig, GeneratorConfig
from steering_factory import model_utils, extraction, sweep, metrics
from steering_factory.storage import VectorStore
from steering_factory.examples_store import ExamplesStore
from steering_factory.generators import build_generator

st.set_page_config(page_title="Steering Vector Factory", layout="wide")

# ---------------------------------------------------------------------------
# Config + cached resources
# ---------------------------------------------------------------------------

cfg = load_config()


@st.cache_resource(show_spinner="Loading target model (first load can take a while)...")
def _load_target_model(name_or_path, dtype, quantization, device_map, layer_override, trust_remote_code):
    tm_cfg = TargetModelConfig(
        name_or_path=name_or_path,
        dtype=dtype,
        quantization=quantization,
        device_map=device_map,
        layer_path_override=layer_override or None,
        trust_remote_code=trust_remote_code,
    )
    return model_utils.load_model(tm_cfg)


@st.cache_resource(show_spinner="Initializing generator backend...")
def _build_generator_cached(backend, anthropic_model, oc_base_url, oc_model, oc_key_env,
                             local_hf_path, local_hf_dtype, local_hf_quant, _loaded_model_id):
    """_loaded_model_id is unused except to bust the cache when the target
    model changes (Streamlit cache_resource hashes args, and a LoadedModel
    object isn't easily hashable, so we key on a stable id instead and look
    the actual object up from session_state)."""
    gen_cfg = GeneratorConfig(
        backend=backend,
        anthropic=cfg.generator.anthropic.__class__(model=anthropic_model),
        openai_compat=cfg.generator.openai_compat.__class__(
            base_url=oc_base_url, model=oc_model, api_key_env=oc_key_env
        ),
        local_hf=cfg.generator.local_hf.__class__(
            name_or_path=local_hf_path or None, dtype=local_hf_dtype, quantization=local_hf_quant
        ),
    )
    loaded_for_local = st.session_state.get("loaded_model") if backend == "local_hf" else None
    return build_generator(gen_cfg, loaded_model_for_local_hf=loaded_for_local)


def get_vector_store() -> VectorStore:
    return VectorStore(cfg.storage.vectors_dir)


def get_examples_store() -> ExamplesStore:
    return ExamplesStore(cfg.storage.examples_dir)


# ---------------------------------------------------------------------------
# Sidebar: model + generator config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Target model")
    name_or_path = st.text_input("Model name or local path", value=cfg.target_model.name_or_path)
    dtype = st.selectbox("dtype", ["bfloat16", "float16", "float32"],
                          index=["bfloat16", "float16", "float32"].index(cfg.target_model.dtype))
    quantization = st.selectbox("Quantization", ["none", "4bit", "8bit"],
                                 index=["none", "4bit", "8bit"].index(cfg.target_model.quantization))
    device_map = st.text_input("device_map", value=cfg.target_model.device_map)
    layer_override = st.text_input(
        "Layer path override (optional)",
        value=cfg.target_model.layer_path_override or "",
        help="Dotted attribute path to the ModuleList of decoder blocks, "
             "e.g. 'model.layers'. Leave blank to auto-detect.",
    )
    trust_remote_code = st.checkbox("trust_remote_code", value=cfg.target_model.trust_remote_code)

    if st.button("Load / reload model", type="primary"):
        st.session_state["loaded_model"] = _load_target_model(
            name_or_path, dtype, quantization, device_map, layer_override, trust_remote_code
        )
        st.session_state["loaded_model_id"] = f"{name_or_path}|{dtype}|{quantization}|{layer_override}"

    loaded = st.session_state.get("loaded_model")
    if loaded:
        st.success(f"Loaded: {name_or_path}\n\n{loaded.num_layers} layers @ `{loaded.layer_path}`, "
                    f"hidden_size={loaded.hidden_size}, device={loaded.device}")
    else:
        st.info("Model not loaded yet.")

    st.divider()
    st.header("Generator backend")
    backend = st.selectbox("Backend", ["anthropic", "openai_compat", "local_hf"],
                            index=["anthropic", "openai_compat", "local_hf"].index(cfg.generator.backend))
    anthropic_model = oc_base_url = oc_model = oc_key_env = local_hf_path = ""
    local_hf_dtype, local_hf_quant = "bfloat16", "4bit"
    if backend == "anthropic":
        anthropic_model = st.text_input("Anthropic model", value=cfg.generator.anthropic.model)
        st.caption("Reads ANTHROPIC_API_KEY from your environment.")
    elif backend == "openai_compat":
        oc_base_url = st.text_input("base_url", value=cfg.generator.openai_compat.base_url)
        oc_model = st.text_input("model", value=cfg.generator.openai_compat.model)
        oc_key_env = st.text_input("API key env var", value=cfg.generator.openai_compat.api_key_env)
    else:
        local_hf_path = st.text_input(
            "Generator model path (blank = reuse target model)",
            value=cfg.generator.local_hf.name_or_path or "",
        )
        local_hf_dtype = st.selectbox("Generator dtype", ["bfloat16", "float16", "float32"], index=0)
        local_hf_quant = st.selectbox("Generator quantization", ["none", "4bit", "8bit"], index=1)

    if st.button("Initialize generator"):
        try:
            st.session_state["generator"] = _build_generator_cached(
                backend, anthropic_model, oc_base_url, oc_model, oc_key_env,
                local_hf_path, local_hf_dtype, local_hf_quant,
                st.session_state.get("loaded_model_id", "none"),
            )
            st.success(f"Generator backend '{backend}' ready.")
        except Exception as e:
            st.error(f"Failed to initialize generator: {e}")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

st.title("Steering Vector Factory")
st.caption(
    "Topic -> contrastive pairs -> extract -> sweep -> save. Treat every saved "
    "vector as a hypothesis, not a finished artifact -- check the "
    "convergence and held-out spectrum before you trust it."
)

tab_examples, tab_extract, tab_sweep, tab_library = st.tabs(
    ["1. Topic & Examples", "2. Extract Vector", "3. Spectrum / Sweep", "4. Vector Library"]
)

# --- Tab 1: Topic & Examples -----------------------------------------------
with tab_examples:
    st.subheader("Define a topic / persona / behavior")
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Topic", value=st.session_state.get("topic", ""),
                               placeholder="e.g. 'terse, no-fluff technical answers'")
        persona = st.text_input("Persona / voice (optional)", value="")
    with col2:
        behavior_description = st.text_area(
            "Behavior description (used by the judge later, can be more precise than the topic)",
            value="", height=80,
        )
        n_pairs = st.number_input("Number of pairs to generate", min_value=4, max_value=200,
                                   value=cfg.defaults.num_pairs_per_topic, step=4)

    gcol1, gcol2 = st.columns([1, 1])
    with gcol1:
        if st.button("Generate pairs with active generator backend"):
            gen = st.session_state.get("generator")
            if not gen:
                st.error("Initialize a generator backend in the sidebar first.")
            elif not topic:
                st.error("Set a topic first.")
            else:
                with st.spinner("Generating contrastive pairs..."):
                    try:
                        pairs = gen.generate_pairs(topic, persona or None, int(n_pairs), behavior_description or None)
                        st.session_state["pairs"] = pairs
                        st.session_state["topic"] = topic
                        st.success(f"Generated {len(pairs)} pairs. Edit them below before extracting.")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
    with gcol2:
        saved_topics = get_examples_store().list_topics()
        load_choice = st.selectbox("...or load a saved example set", [""] + saved_topics)
        if load_choice and st.button("Load"):
            data = get_examples_store().load(load_choice)
            if data:
                st.session_state["pairs"] = data["pairs"]
                st.session_state["topic"] = data["topic"]
                st.success(f"Loaded {len(data['pairs'])} pairs for '{load_choice}'.")

    st.divider()
    st.subheader("Edit pairs")
    pairs = st.session_state.get("pairs", [])
    if pairs:
        df = pd.DataFrame(pairs)
        edited = st.data_editor(
            df, num_rows="dynamic", use_container_width=True, height=400,
            column_config={
                "prompt": st.column_config.TextColumn(width="medium"),
                "compliant": st.column_config.TextColumn(width="large"),
                "non_compliant": st.column_config.TextColumn(width="large"),
            },
        )
        st.session_state["pairs"] = edited.to_dict("records")

        if st.button("Save this example set to disk"):
            get_examples_store().save(
                st.session_state.get("topic", topic), st.session_state["pairs"],
                persona=persona or None, generator_backend=backend,
            )
            st.success("Saved to data/examples/.")
    else:
        st.info("No pairs yet -- generate some above, or load a saved set.")

# --- Tab 2: Extract Vector --------------------------------------------------
with tab_extract:
    st.subheader("Extract a steering vector")
    loaded = st.session_state.get("loaded_model")
    pairs = st.session_state.get("pairs", [])

    if not loaded:
        st.warning("Load a target model in the sidebar first.")
    elif not pairs:
        st.warning("Add contrastive pairs in tab 1 first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            layer_fraction = st.slider("Layer depth (fraction)", 0.0, 1.0, cfg.defaults.layer_fraction, 0.05)
            layer_idx = model_utils.layer_index_from_fraction(loaded.num_layers, layer_fraction)
            st.caption(f"Resolved layer index: {layer_idx} / {loaded.num_layers - 1}")
        with col2:
            method = st.selectbox("Extraction method", ["mean_diff", "pca", "optimized"],
                                   index=["mean_diff", "pca", "optimized"].index(cfg.defaults.extraction_method))
            pooling = st.selectbox("Pooling", ["last", "mean"], index=0)
        with col3:
            opt_steps, opt_lr, opt_coeff = 40, 0.05, 1.0
            if method == "optimized":
                opt_steps = st.number_input("Optimization steps", 5, 200, 40)
                opt_lr = st.number_input("Learning rate", 0.001, 1.0, 0.05, format="%.3f")
                opt_coeff = st.number_input("Training-time coefficient", 0.1, 10.0, 1.0)
                st.caption("Slow: full forward+backward per pair per step. Start with a small "
                           "pair subset if iterating.")

        if st.button("Extract vector", type="primary"):
            with st.spinner(f"Extracting via {method}..."):
                try:
                    kwargs = {}
                    if method == "optimized":
                        kwargs = dict(steps=int(opt_steps), lr=float(opt_lr), coefficient=float(opt_coeff))
                    result = extraction.extract(method, loaded, layer_idx, pairs, pooling=pooling, **kwargs)
                    st.session_state["extraction_result"] = result
                    st.success(f"Extracted a {method} vector at layer {layer_idx} from {len(pairs)} pairs.")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    raise

        result = st.session_state.get("extraction_result")
        if result:
            st.divider()
            st.markdown(f"**Active vector:** method=`{result.method}`, layer=`{result.layer_idx}`, "
                        f"pooling=`{result.pooling}`, n_pairs=`{result.num_pairs}`, "
                        f"norm=`{result.vector.norm().item():.3f}`")

            if result.method != "optimized" and result.convergence:
                conv_df = pd.DataFrame(result.convergence)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=conv_df["n"], y=conv_df["cos_to_final"],
                                          mode="lines+markers", name="cosine sim to final vector"))
                fig.update_layout(title="Convergence: does the vector direction stabilize as N grows?",
                                   xaxis_title="number of pairs used", yaxis_title="cosine similarity to final vector",
                                   yaxis_range=[-1, 1], height=350)
                st.plotly_chart(fig, use_container_width=True)
                summary = metrics.convergence_summary(result.convergence)
                st.info(summary["note"])
            elif result.method == "optimized" and result.convergence:
                conv_df = pd.DataFrame(result.convergence)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=conv_df["n"], y=conv_df["loss"], mode="lines", name="training loss"))
                fig.update_layout(title="Optimization loss (lower = better separation of compliant/non_compliant)",
                                   xaxis_title="step", yaxis_title="loss", height=350)
                st.plotly_chart(fig, use_container_width=True)
                st.warning("Optimized vectors don't get the same N-pairs convergence curve -- "
                           "cross-check this one against the held-out spectrum in tab 3 before trusting it.")

# --- Tab 3: Spectrum / Sweep -------------------------------------------------
with tab_sweep:
    st.subheader("Coefficient sweep on a test prompt")
    loaded = st.session_state.get("loaded_model")
    result = st.session_state.get("extraction_result")

    if not loaded or not result:
        st.warning("Load a model and extract (or load from the library) a vector first.")
    else:
        test_prompt = st.text_area(
            "Test prompt (ideally NOT one of your training pairs -- this is your held-out check)",
            value="", height=80,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            coeff_min = st.number_input("Min coefficient", value=-4.0)
            coeff_max = st.number_input("Max coefficient", value=4.0)
        with col2:
            coeff_steps = st.number_input("Number of steps", 3, 25, 9)
            max_new_tokens = st.number_input("Max new tokens", 16, 512, cfg.defaults.max_new_tokens)
        with col3:
            fluency_cap = st.number_input("Fluency cap (perplexity ratio)", 1.0, 5.0,
                                           cfg.defaults.fluency_cap_ratio)
            use_judge = st.checkbox("Use generator backend as behavior judge", value=False)

        coefficients = [
            round(coeff_min + i * (coeff_max - coeff_min) / (coeff_steps - 1), 3)
            for i in range(int(coeff_steps))
        ] if coeff_steps > 1 else [coeff_min]

        behavior_desc_for_judge = st.session_state.get("topic", "")
        if use_judge:
            behavior_desc_for_judge = st.text_input(
                "Behavior description for judge", value=behavior_desc_for_judge
            )

        if st.button("Run sweep", type="primary"):
            if not test_prompt.strip():
                st.error("Enter a test prompt.")
            else:
                judge = st.session_state.get("generator") if use_judge else None
                with st.spinner(f"Generating at {len(coefficients)} coefficients..."):
                    try:
                        sweep_out = sweep.coefficient_sweep(
                            loaded, result.layer_idx, result.vector, test_prompt, coefficients,
                            max_new_tokens=int(max_new_tokens), judge=judge,
                            behavior_description=behavior_desc_for_judge if use_judge else None,
                        )
                        st.session_state["sweep_result"] = sweep_out
                        st.session_state["sweep_prompt"] = test_prompt
                    except Exception as e:
                        st.error(f"Sweep failed: {e}")
                        raise

        sweep_out = st.session_state.get("sweep_result")
        if sweep_out:
            rows = sweep_out["rows"]
            df = pd.DataFrame(rows)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["coefficient"], y=df["perplexity_ratio"],
                                      mode="lines+markers", name="perplexity ratio (fluency cost)"))
            fig.add_trace(go.Scatter(x=df["coefficient"], y=df["js_divergence_vs_baseline"],
                                      mode="lines+markers", name="JS divergence vs baseline", yaxis="y2"))
            if "behavior_score" in df.columns and df["behavior_score"].notna().any():
                fig.add_trace(go.Scatter(x=df["coefficient"], y=df["behavior_score"],
                                          mode="lines+markers", name="judge behavior score", yaxis="y2"))
            fig.update_layout(
                title="Spectrum: fluency cost vs. behavioral shift across coefficients",
                xaxis_title="coefficient", yaxis_title="perplexity ratio",
                yaxis2=dict(title="divergence / behavior score (0-1ish)", overlaying="y", side="right"),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            best_pos, best_neg = sweep.suggest_best_coefficients(rows, fluency_cap_ratio=fluency_cap)
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                st.markdown("**Suggested positive (amplify) coefficient**")
                st.write(best_pos if best_pos else "None within fluency cap")
            with bcol2:
                st.markdown("**Suggested negative (suppress/refuse) coefficient**")
                st.write(best_neg if best_neg else "None within fluency cap")

            st.divider()
            st.markdown("**Outputs at each coefficient**")
            for row in rows:
                marker = " (baseline)" if row["coefficient"] == 0 else ""
                label = f"coefficient = {row['coefficient']}{marker}"
                if row["perplexity_ratio"]:
                    label += f" | ppl_ratio={row['perplexity_ratio']:.2f} | js={row['js_divergence_vs_baseline']:.3f}"
                with st.expander(label):
                    st.write(row["text"])
                    extra = f"repetition_score={row['repetition_score']:.3f}, distinct_2={row['distinct_2']:.3f}"
                    if row.get("behavior_score") is not None:
                        extra += f", behavior_score={row['behavior_score']:.3f}"
                    st.caption(extra)

            st.divider()
            st.subheader("Save this vector")
            notes = st.text_input("Notes (optional)", value="")
            if st.button("Save vector to local library", type="primary"):
                store = get_vector_store()
                conv_summary = (
                    metrics.convergence_summary(result.convergence)
                    if result.method != "optimized" else None
                )
                record = store.save(
                    topic=st.session_state.get("topic", "untitled"),
                    model_name=name_or_path,
                    layer_idx=result.layer_idx,
                    method=result.method,
                    pooling=result.pooling,
                    num_pairs=result.num_pairs,
                    vector=result.vector,
                    best_positive_coefficient=best_pos["coefficient"] if best_pos else None,
                    best_negative_coefficient=best_neg["coefficient"] if best_neg else None,
                    convergence_summary=conv_summary,
                    sweep_summary=rows,
                    notes=notes,
                    extra={"test_prompt": st.session_state.get("sweep_prompt", "")},
                )
                st.success(f"Saved vector `{record.id}` for topic '{record.topic}' "
                           f"to {cfg.storage.vectors_dir}/{record.vector_file}")

# --- Tab 4: Vector Library ----------------------------------------------------
with tab_library:
    st.subheader("Saved vectors")
    store = get_vector_store()
    topic_filter = st.text_input("Filter by topic contains", value="")
    records = store.list(topic_filter=topic_filter or None)

    if not records:
        st.info("No saved vectors yet.")
    else:
        for r in records:
            with st.expander(f"{r['topic']}  |  {r['method']} @ layer {r['layer_idx']}  |  "
                              f"{r['model_name']}  |  id={r['id']}"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.json({k: v for k, v in r.items() if k != "sweep_summary"}, expanded=False)
                with c2:
                    if st.button("Load into workspace", key=f"load_{r['id']}"):
                        vec = store.load_vector(r["id"])
                        loaded_obj = type("LoadedVectorRef", (), {})()
                        loaded_obj.vector = vec
                        loaded_obj.method = r["method"]
                        loaded_obj.layer_idx = r["layer_idx"]
                        loaded_obj.pooling = r["pooling"]
                        loaded_obj.num_pairs = r["num_pairs"]
                        loaded_obj.convergence = []
                        st.session_state["extraction_result"] = loaded_obj
                        st.session_state["topic"] = r["topic"]
                        st.success(f"Loaded vector {r['id']} into the active workspace -- "
                                   f"go to tab 3 to sweep it against a new prompt.")
                    if st.button("Delete", key=f"del_{r['id']}"):
                        store.delete(r["id"])
                        st.rerun()
