from pathlib import Path
from tom_cqed_tools.circuit_diagram_builder import CircuitDiagramBuilder

# Initialize with 'transparent' theme to use your default alphas
diagram = CircuitDiagramBuilder(['Transmon', 'Buffer'], theme='transparent')

# 1. State Prep
diagram.pulse(mode='Transmon', label=r'$\sqrt{X_{ge}}$', color='green')
diagram.advance()
diagram.pulse(mode='Transmon', label=r'$X_{ef}$', color='blue', alpha=0.5)
diagram.advance()
diagram.two_mode_gate(mode1='Transmon', mode2='Buffer', label=r'$X_{01}^{SB}$', color='red')
diagram.advance(dt=1.6)

# 3. Evolution
diagram.delay(mode='Buffer', label=r'Delay $(\tau)$')
diagram.advance(dt=1.6) 

# 5. Unload
diagram.two_mode_gate(mode1='Transmon', mode2='Buffer', label=r'$X_{01}^{SB}$', color='red')
diagram.advance()

# Override width specifically for this wide text so it doesn't get squished
diagram.pulse(mode='Transmon', label=r'$X_{ef}$', color='blue',alpha=0.5)
diagram.advance()

diagram.pulse(mode='Transmon', label=r'$\sqrt{X_{ge}^\varphi}$', color='green', width=1.0)
diagram.advance()

# 6. Readout
diagram.measure(mode=['Transmon'])

# 7. Portable Path Resolution
try:
    save_dir = Path(__file__).resolve().parent
except NameError:
    save_dir = Path.cwd() / "figure-making"

save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "buffer_ramsey_protocol.pdf"

diagram.render(save_as=save_path)
