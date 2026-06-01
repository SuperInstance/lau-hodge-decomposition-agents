//! Agent learning application of Hodge decomposition.
//!
//! Every agent signal decomposes into:
//! - Exact (exploration): what the agent learned through gradient descent
//! - Coexact (exploitation): what the agent was told through instruction/signals
//! - Harmonic (prior knowledge): what the agent already knew (topological invariants)
//!
//! This gives a principled decomposition of agent behavior into
//! three orthogonal components with distinct semantic interpretations.

use serde::{Serialize, Deserialize};
use crate::forms::{DifferentialForm, SimplicialComplex};
use crate::decomposition::{decompose, HodgeDecomposition};
use crate::betti::betti_number;
use crate::spectral::SpectralAnalysis;
use crate::laplacian::is_harmonic;

/// An agent in the system, represented as a signal over a simplicial complex.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Agent {
    /// Agent identifier
    pub id: String,
    /// The agent's current signal (a k-form)
    pub signal: DifferentialForm,
    /// Semantic labels for each component of the signal
    pub labels: Vec<String>,
}

impl Agent {
    /// Create a new agent.
    pub fn new(id: impl Into<String>, signal: DifferentialForm) -> Self {
        Self {
            id: id.into(),
            labels: vec![],
            signal,
        }
    }

    /// Decompose the agent's signal into exploration + exploitation + prior.
    pub fn analyze(&self, complex: &SimplicialComplex) -> AgentAnalysis {
        let decomp = decompose(complex, &self.signal);
        let total_energy = self.signal.norm_squared();

        let exploration_energy = decomp.exact.norm_squared();
        let exploitation_energy = decomp.coexact.norm_squared();
        let prior_energy = decomp.harmonic.norm_squared();

        AgentAnalysis {
            agent_id: self.id.clone(),
            exploration: decomp.exact.clone(),
            exploitation: decomp.coexact.clone(),
            prior_knowledge: decomp.harmonic.clone(),
            exploration_ratio: if total_energy > 0.0 { exploration_energy / total_energy } else { 0.0 },
            exploitation_ratio: if total_energy > 0.0 { exploitation_energy / total_energy } else { 0.0 },
            prior_ratio: if total_energy > 0.0 { prior_energy / total_energy } else { 0.0 },
            total_energy,
        }
    }
}

/// Analysis result for an agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentAnalysis {
    pub agent_id: String,
    /// Exact component (exploration)
    pub exploration: DifferentialForm,
    /// Coexact component (exploitation)
    pub exploitation: DifferentialForm,
    /// Harmonic component (prior knowledge)
    pub prior_knowledge: DifferentialForm,
    /// Fraction of signal that is exploration
    pub exploration_ratio: f64,
    /// Fraction of signal that is exploitation
    pub exploitation_ratio: f64,
    /// Fraction of signal that is prior knowledge
    pub prior_ratio: f64,
    /// Total signal energy
    pub total_energy: f64,
}

impl AgentAnalysis {
    /// Classify the agent's dominant mode.
    pub fn dominant_mode(&self) -> AgentMode {
        if self.exploration_ratio >= self.exploitation_ratio
            && self.exploration_ratio >= self.prior_ratio
        {
            AgentMode::Exploration
        } else if self.exploitation_ratio >= self.prior_ratio {
            AgentMode::Exploitation
        } else {
            AgentMode::PriorKnowledge
        }
    }

    /// Is the agent primarily in exploration mode?
    pub fn is_exploring(&self) -> bool {
        matches!(self.dominant_mode(), AgentMode::Exploration)
    }

    /// Is the agent primarily in exploitation mode?
    pub fn is_exploiting(&self) -> bool {
        matches!(self.dominant_mode(), AgentMode::Exploitation)
    }

    /// How "balanced" is the agent? (entropy of the decomposition)
    pub fn balance(&self) -> f64 {
        let ratios = [self.exploration_ratio, self.exploitation_ratio, self.prior_ratio];
        let entropy: f64 = ratios.iter()
            .filter(|&&r| r > 0.0)
            .map(|&r| -r * r.ln())
            .sum();
        entropy / (3.0_f64.ln()) // Normalize to [0, 1]
    }
}

/// The dominant mode of an agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentMode {
    /// Exact component dominant: agent is learning through exploration
    Exploration,
    /// Coexact component dominant: agent is following instruction/exploitation
    Exploitation,
    /// Harmonic component dominant: agent is relying on prior knowledge
    PriorKnowledge,
}

impl std::fmt::Display for AgentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentMode::Exploration => write!(f, "Exploration"),
            AgentMode::Exploitation => write!(f, "Exploitation"),
            AgentMode::PriorKnowledge => write!(f, "PriorKnowledge"),
        }
    }
}

/// A system of agents interacting over a shared topological space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentSystem {
    /// The simplicial complex representing the shared space
    pub complex: SimplicialComplex,
    /// The agents in the system
    pub agents: Vec<Agent>,
    /// The degree of forms being analyzed
    pub form_degree: usize,
}

impl AgentSystem {
    /// Create a new agent system.
    pub fn new(complex: SimplicialComplex, form_degree: usize) -> Self {
        Self {
            complex,
            agents: vec![],
            form_degree,
        }
    }

    /// Add an agent to the system.
    pub fn add_agent(&mut self, agent: Agent) {
        self.agents.push(agent);
    }

    /// Analyze all agents.
    pub fn analyze_all(&self) -> Vec<AgentAnalysis> {
        self.agents.iter().map(|a| a.analyze(&self.complex)).collect()
    }

    /// Compute the average exploration ratio across all agents.
    pub fn avg_exploration(&self) -> f64 {
        let analyses = self.analyze_all();
        if analyses.is_empty() { return 0.0; }
        analyses.iter().map(|a| a.exploration_ratio).sum::<f64>() / analyses.len() as f64
    }

    /// Compute the average exploitation ratio.
    pub fn avg_exploitation(&self) -> f64 {
        let analyses = self.analyze_all();
        if analyses.is_empty() { return 0.0; }
        analyses.iter().map(|a| a.exploitation_ratio).sum::<f64>() / analyses.len() as f64
    }

    /// Compute the average prior knowledge ratio.
    pub fn avg_prior(&self) -> f64 {
        let analyses = self.analyze_all();
        if analyses.is_empty() { return 0.0; }
        analyses.iter().map(|a| a.prior_ratio).sum::<f64>() / analyses.len() as f64
    }

    /// Get the topological complexity (number of "holes" in the space).
    pub fn topological_complexity(&self) -> usize {
        (0..=self.complex.dimension)
            .map(|k| betti_number(&self.complex, k))
            .sum()
    }

    /// Spectral analysis of the shared space.
    pub fn spectral_profile(&self) -> Vec<SpectralAnalysis> {
        (0..=self.complex.dimension)
            .map(|k| SpectralAnalysis::analyze(&self.complex, k))
            .collect()
    }

    /// Compute the "knowledge diversity" — how different agents'
    /// decomposition profiles are.
    pub fn knowledge_diversity(&self) -> f64 {
        let analyses = self.analyze_all();
        if analyses.len() < 2 { return 0.0; }

        let mut total_dist = 0.0;
        let mut count = 0;
        for i in 0..analyses.len() {
            for j in i+1..analyses.len() {
                let di = (analyses[i].exploration_ratio, analyses[i].exploitation_ratio, analyses[i].prior_ratio);
                let dj = (analyses[j].exploration_ratio, analyses[j].exploitation_ratio, analyses[j].prior_ratio);
                let dist = ((di.0 - dj.0).powi(2) + (di.1 - dj.1).powi(2) + (di.2 - dj.2).powi(2)).sqrt();
                total_dist += dist;
                count += 1;
            }
        }
        total_dist / count as f64
    }

    /// Recommend the optimal learning strategy for the system.
    pub fn recommend_strategy(&self) -> LearningStrategy {
        let exp = self.avg_exploration();
        let exl = self.avg_exploitation();
        let pri = self.avg_prior();

        if pri > 0.7 {
            LearningStrategy::LeveragePriors
        } else if exp > exl {
            LearningStrategy::MoreExploitation
        } else if exl > exp {
            LearningStrategy::MoreExploration
        } else {
            LearningStrategy::Balanced
        }
    }
}

/// Learning strategy recommendations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningStrategy {
    /// Agents should explore more
    MoreExploration,
    /// Agents should exploit more
    MoreExploitation,
    /// Leverage existing prior knowledge
    LeveragePriors,
    /// Current balance is good
    Balanced,
}

impl std::fmt::Display for LearningStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LearningStrategy::MoreExploration => write!(f, "MoreExploration"),
            LearningStrategy::MoreExploitation => write!(f, "MoreExploitation"),
            LearningStrategy::LeveragePriors => write!(f, "LeveragePriors"),
            LearningStrategy::Balanced => write!(f, "Balanced"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_system() -> AgentSystem {
        let sc = SimplicialComplex::triangle();
        let mut system = AgentSystem::new(sc, 0);

        system.add_agent(Agent::new("explorer",
            DifferentialForm::new(0, vec![1.0, 0.0, 0.0])));
        system.add_agent(Agent::new("exploiter",
            DifferentialForm::new(0, vec![1.0, 1.0, 1.0])));
        system.add_agent(Agent::new("mixed",
            DifferentialForm::new(0, vec![1.0, 2.0, 3.0])));

        system
    }

    #[test]
    fn test_agent_creation() {
        let a = Agent::new("test", DifferentialForm::new(0, vec![1.0, 2.0, 3.0]));
        assert_eq!(a.id, "test");
        assert_eq!(a.signal.degree, 0);
    }

    #[test]
    fn test_agent_analysis_components() {
        let sc = SimplicialComplex::triangle();
        let a = Agent::new("test", DifferentialForm::new(0, vec![1.0, 2.0, 3.0]));
        let analysis = a.analyze(&sc);
        assert_eq!(analysis.exploration.degree, 0);
        assert_eq!(analysis.exploitation.degree, 0);
        assert_eq!(analysis.prior_knowledge.degree, 0);
    }

    #[test]
    fn test_agent_analysis_ratios_sum_to_one() {
        let sc = SimplicialComplex::triangle();
        let a = Agent::new("test", DifferentialForm::new(0, vec![1.0, 2.0, 3.0]));
        let analysis = a.analyze(&sc);
        let sum = analysis.exploration_ratio + analysis.exploitation_ratio + analysis.prior_ratio;
        assert!((sum - 1.0).abs() < 0.01, "Ratios sum to {}", sum);
    }

    #[test]
    fn test_constant_agent_is_harmonic() {
        let sc = SimplicialComplex::triangle();
        let a = Agent::new("constant", DifferentialForm::new(0, vec![1.0, 1.0, 1.0]));
        let analysis = a.analyze(&sc);
        assert!(analysis.prior_ratio > 0.9);
    }

    #[test]
    fn test_agent_system_creation() {
        let sys = triangle_system();
        assert_eq!(sys.agents.len(), 3);
    }

    #[test]
    fn test_system_analyze_all() {
        let sys = triangle_system();
        let analyses = sys.analyze_all();
        assert_eq!(analyses.len(), 3);
    }

    #[test]
    fn test_system_avg_ratios() {
        let sys = triangle_system();
        let exp = sys.avg_exploration();
        let exl = sys.avg_exploitation();
        let pri = sys.avg_prior();
        assert!(exp >= 0.0 && exp <= 1.0);
        assert!(exl >= 0.0 && exl <= 1.0);
        assert!(pri >= 0.0 && pri <= 1.0);
    }

    #[test]
    fn test_topological_complexity_triangle() {
        let sys = triangle_system();
        assert_eq!(sys.topological_complexity(), 1); // Just H^0 = 1
    }

    #[test]
    fn test_spectral_profile() {
        let sys = triangle_system();
        let profile = sys.spectral_profile();
        assert_eq!(profile.len(), 3);
    }

    #[test]
    fn test_knowledge_diversity() {
        let sys = triangle_system();
        let div = sys.knowledge_diversity();
        assert!(div >= 0.0);
    }

    #[test]
    fn test_recommend_strategy() {
        let sys = triangle_system();
        let strategy = sys.recommend_strategy();
        // Should return a valid strategy
        let _ = format!("{}", strategy);
    }

    #[test]
    fn test_agent_mode_display() {
        assert_eq!(format!("{}", AgentMode::Exploration), "Exploration");
        assert_eq!(format!("{}", AgentMode::Exploitation), "Exploitation");
        assert_eq!(format!("{}", AgentMode::PriorKnowledge), "PriorKnowledge");
    }

    #[test]
    fn test_learning_strategy_display() {
        assert_eq!(format!("{}", LearningStrategy::Balanced), "Balanced");
    }

    #[test]
    fn test_agent_balance() {
        let sc = SimplicialComplex::triangle();
        let a = Agent::new("balanced",
            DifferentialForm::new(0, vec![1.0, 2.0, 3.0]));
        let analysis = a.analyze(&sc);
        let balance = analysis.balance();
        assert!(balance >= 0.0 && balance <= 1.0);
    }

    #[test]
    fn test_agent_dominant_mode() {
        let sc = SimplicialComplex::triangle();
        let a = Agent::new("constant", DifferentialForm::new(0, vec![1.0, 1.0, 1.0]));
        let analysis = a.analyze(&sc);
        assert_eq!(analysis.dominant_mode(), AgentMode::PriorKnowledge);
    }

    #[test]
    fn test_1form_agent_analysis() {
        let sc = SimplicialComplex::triangle();
        let a = Agent::new("1form_agent",
            DifferentialForm::new(1, vec![1.0, 2.0, 3.0]));
        let analysis = a.analyze(&sc);
        assert_eq!(analysis.exploration.degree, 1);
        assert_eq!(analysis.exploitation.degree, 1);
        assert_eq!(analysis.prior_knowledge.degree, 1);
    }

    #[test]
    fn test_tetrahedron_agent_system() {
        let sc = SimplicialComplex::tetrahedron();
        let mut sys = AgentSystem::new(sc, 1);
        sys.add_agent(Agent::new("t1",
            DifferentialForm::new(1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
        sys.add_agent(Agent::new("t2",
            DifferentialForm::new(1, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0])));
        let analyses = sys.analyze_all();
        assert_eq!(analyses.len(), 2);
    }

    #[test]
    fn test_agent_serialization() {
        let a = Agent::new("test", DifferentialForm::new(0, vec![1.0, 2.0]));
        let json = serde_json::to_string(&a).unwrap();
        let decoded: Agent = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "test");
    }

    #[test]
    fn test_analysis_serialization() {
        let sc = SimplicialComplex::triangle();
        let a = Agent::new("test", DifferentialForm::new(0, vec![1.0, 2.0, 3.0]));
        let analysis = a.analyze(&sc);
        let json = serde_json::to_string(&analysis).unwrap();
        assert!(json.contains("test"));
    }

    #[test]
    fn test_empty_agent_system() {
        let sc = SimplicialComplex::triangle();
        let sys = AgentSystem::new(sc, 0);
        assert_eq!(sys.agents.len(), 0);
        assert_eq!(sys.avg_exploration(), 0.0);
    }
}
