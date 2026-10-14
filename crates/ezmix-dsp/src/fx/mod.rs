//! Effect implementations, one per engine effect ID.

pub mod amps;
pub mod delays;
pub mod dist;
pub mod dynamics;
pub mod eq;
pub mod ir;
pub mod modulation;
pub mod reverbs;

use crate::effect::{Bypass, Effect};

/// Instantiate the processor for an engine effect ID. Returns `None` for IDs
/// with no implementation (the slot is then bypassed).
pub fn create(id: &str) -> Option<Box<dyn Effect>> {
    use amps::*;
    use delays::*;
    use dist::*;
    use dynamics::*;
    use eq::*;
    use modulation::*;
    use reverbs::*;

    let fx: Box<dyn Effect> = match id {
        "TA_DUMMY" => Box::new(Bypass),
        // EQ / filters
        "TA_TOONTRACK_EQ2" => Box::new(ParametricEq::new()),
        "TA_TOONTRACK_EQ2_MS" => Box::new(ParametricEqMs::new()),
        "TA_TOONTRACK_LPFHPF" => Box::new(LpfHpf::new(false)),
        "TA_TOONTRACK_LPFHPF_MS" => Box::new(LpfHpf::new(true)),
        "TA_EXTENDED_EQ" => Box::new(ExtendedEq::new()),
        "TA_EQ84" => Box::new(Eq84::new()),
        "TA_EQP1A" => Box::new(Pultec::new()),
        "TA_EQ550A" => Box::new(Api550::new()),
        "TA_TTFILTER" => Box::new(CreativeFilter::new()),
        "TA_SRSWOW" => Box::new(WowEnhancer::new()),
        "TA_TT_MATCHINGEQ" => Box::new(ir::MatchingEq::new()),
        // Dynamics
        "TA_TOONTRACK_COMPR" | "TA_SLKSCOMPRESSOR" => Box::new(Compressor::new(false)),
        "TA_TOONTRACK_COMPRMS" => Box::new(Compressor::new(true)),
        "TA_TOONTRACK_LIMITER" => Box::new(Limiter::new(false)),
        "TA_TOONTRACK_LIMITERMS" => Box::new(Limiter::new(true)),
        "TA_MASTERLIMITER" | "TA_MASTERLIMITERTHRESHOLD" => Box::new(MasterLimiter::new()),
        "TA_MULTIBANDCOMPR" => Box::new(MultibandComp::new(false)),
        "TA_MULTIBANDCOMPRTHRESHOLD" => Box::new(MultibandComp::new(true)),
        "TA_1176" => Box::new(Fet::new()),
        "TA_LA2A" => Box::new(Opto::new(false)),
        "TA_LA3A" => Box::new(Opto::new(true)),
        "TA_FAIRCHILD670" => Box::new(VariMu::new()),
        "TA_SSLCOMPG" => Box::new(VcaBus::new()),
        "TA_SSLLMC" => Box::new(ListenMic::new()),
        "TA_DBX160" => Box::new(Dbx160::new()),
        "TA_LIMITER143" => Box::new(PedalLimiter::new()),
        "TA_TOONTRACK_OVERLOUD" => Box::new(OneKnob::new()),
        "TA_DBX180A" => Box::new(Compander::new(CompanderKind::Punch)),
        "TA_DOLBY361" => Box::new(Compander::new(CompanderKind::Dolby)),
        "TA_DOLBY361MOD" => Box::new(Compander::new(CompanderKind::DolbyVocal)),
        "TA_DEESSER" => Box::new(DeEsser::new()),
        "TA_FREQ_GATE" => Box::new(Gate::new()),
        "TA_TRANSIENT_DESIGNER" => Box::new(TransientShaper::new()),
        "TA_TTEXCITER" => Box::new(Exciter::new()),
        "TA_EVENHARM" => Box::new(Harmonizer::new()),
        // Drive
        "TA_DIST" => Box::new(Pedal::new(PedalKind::Screamer)),
        "TA_CENTAUR" => Box::new(Pedal::new(PedalKind::Klon)),
        "TA_BIGM" => Box::new(Pedal::new(PedalKind::Muff)),
        "TA_FUZZRACE" => Box::new(Pedal::new(PedalKind::Fuzz)),
        "TA_METALPLANET" => Box::new(Pedal::new(PedalKind::Metal)),
        "TA_CULTURE_VULTURE" => Box::new(ValveBox::new()),
        "TA_TOONTRACK_TAPESIM" => Box::new(Tape::new(false)),
        "TA_TAPEEMU" => Box::new(Tape::new(true)),
        "TA_BIT_CRUSH" => Box::new(BitCrush::new(false)),
        "TA_BIT_CRUSH2" => Box::new(BitCrush::new(true)),
        "TA_MULTIBANDDIST" => Box::new(MultibandDist::new()),
        "TA_CONSOLEEMU" => Box::new(Console::new()),
        "TA_TCPRE" => Box::new(Preamp::new(PreampKind::TcPre)),
        "TA_RHODESPRE" => Box::new(Preamp::new(PreampKind::Rhodes)),
        "TA_T21BASSDRIVE" => Box::new(Preamp::new(PreampKind::BassDriver)),
        "TA_ALPHAOMEGA" => Box::new(Preamp::new(PreampKind::AlphaOmega)),
        // Amps / cabs
        "TA_GUITAR_AMP" => Box::new(Amp::new(AmpKind::Guitar)),
        "TA_SVT_CL" => Box::new(Amp::new(AmpKind::SvtClassic)),
        "TA_TWEED_DELUXE" => Box::new(Amp::new(AmpKind::TweedDeluxe)),
        "TA_MESA_50CAL_CLEAN" => Box::new(Amp::new(AmpKind::Mesa50Clean)),
        "TA_MESA_50CAL_LEAD" => Box::new(Amp::new(AmpKind::Mesa50Lead)),
        "TA_LABSERIES5CLEAN" => Box::new(Amp::new(AmpKind::LabSeriesClean)),
        "TA_LABSERIES5DIST" => Box::new(Amp::new(AmpKind::LabSeriesDist)),
        "TA_MSUPERLEAD" | "TA_MSUPERBASS" => Box::new(Amp::new(AmpKind::SuperLead)),
        "TA_BASSAMP1" | "TA_BASSAMP2" | "TA_BASSAMP3" => Box::new(Amp::new(AmpKind::BassTube)),
        "TA_AMPEG_B15N" => Box::new(Amp::new(AmpKind::AmpegB15)),
        "TA_BASS8X10" => Box::new(BassCab::new(true)),
        "TA_BASS1x15" => Box::new(BassCab::new(false)),
        "TA_CABIR" => Box::new(ir::CabIr::new()),
        "TA_LESLIE" | "TA_LESLIE2" => Box::new(Rotary::new(0)),
        "TA_LESLIE3" => Box::new(Rotary::new(1)),
        // Modulation
        "TA_TOONTRACK_CHORUS" => Box::new(Chorus::new(ChorusKind::Basic)),
        "TA_SOLINA_MOD" => Box::new(Chorus::new(ChorusKind::Solina)),
        "TA_TTTRICHORUS" => Box::new(Chorus::new(ChorusKind::Tri)),
        "TA_DIMENSIOND" => Box::new(Chorus::new(ChorusKind::Dimension)),
        "TA_CEB3" => Box::new(Chorus::new(ChorusKind::BassChorus)),
        "TA_FLANGER" => Box::new(Flanger::new()),
        "TA_PHASE90" => Box::new(Phaser::new(false)),
        "TA_TTSMALLSTONE" => Box::new(Phaser::new(true)),
        "TA_TREMOLO" => Box::new(Tremolo::new()),
        "TA_VB2" => Box::new(Vibrato::new(false)),
        "TA_HAMMONDVIBRATO" | "TA_HAMMONDVIBRATO2" => Box::new(Vibrato::new(true)),
        "TA_AUTOWAH" => Box::new(AutoWah::new()),
        "TA_OCTAVER2" => Box::new(Octaver::new()),
        "TA_POG2" => Box::new(Pog::new()),
        "TA_TTSPACER" => Box::new(StereoEnhancer::new()),
        // Delays
        "TA_ECHOTAPE" => Box::new(TapeDelay::new(false)),
        "TA_ECHOFILTER" => Box::new(TapeDelay::new(true)),
        "TA_AMSRMX16ECHO" => Box::new(DigitalEcho::new()),
        "TA_ADASTD1" => Box::new(TappedDelay::new()),
        "TA_ECHOSON" => Box::new(DrumEcho::new()),
        // Reverbs
        "TA_REVHALL" => Box::new(Reverb::new(ReverbKind::Hall)),
        "TA_REVROOM" => Box::new(Reverb::new(ReverbKind::Room)),
        "TA_REVPLATE" => Box::new(Reverb::new(ReverbKind::Plate)),
        "TA_REVSPRING" => Box::new(Reverb::new(ReverbKind::Spring)),
        "TA_REVINVERSE" => Box::new(SparseReverb::new(true)),
        "TA_AMSRMX16NONLIN" => Box::new(SparseReverb::new(false)),
        "TA_REVIR" => Box::new(ir::ReverbIr::new()),
        _ => return None,
    };
    Some(fx)
}
