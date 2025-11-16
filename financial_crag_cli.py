"""
Financial CRAG CLI System
"""

import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from financial_crag import FinancialCRAG

console = Console()


class CRAGCLI:
    """CRAG CLI"""

    def __init__(self):
        self.system = None
        self.ticker = None

    def banner(self):
        console.print("\n[bold cyan]💹 FINANCIAL CRAG SYSTEM[/bold cyan]")
        console.print("[dim]Commands: setup <TICKER> | query <Q> | help | exit[/dim]\n")

    def help(self):
        console.print(Panel("""
        [yellow]setup TICKER[/yellow]  - Load stock (e.g., setup AAPL)
        [yellow]query QUESTION[/yellow] - Ask about stock
        [yellow]help[/yellow]          - Show this
        [yellow]exit[/yellow]          - Quit
        
        [cyan]Examples:[/cyan]
          • What is the P/E ratio?
          • Why did the stock move today?
          • Recent news?
                """, title="Help"))

    def setup(self, ticker: str) -> bool:
        ticker = ticker.upper().strip()
        if not ticker:
            console.print("[red]❌ Provide ticker[/red]")
            return False

        try:
            with console.status(f"[green]Loading {ticker}...", spinner="dots"):
                if not self.system:
                    self.system = FinancialCRAG()
                self.system.setup(ticker)
                self.ticker = ticker

            console.print(f"[green]✅ {ticker} ready![/green]\n")
            return True
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return False

    def query(self, question: str):
        if not self.system or not self.ticker:
            console.print("[yellow]⚠️ Run 'setup <TICKER>' first[/yellow]")
            return

        if not question:
            console.print("[red]❌ Provide question[/red]")
            return

        try:
            with console.status("[green]Thinking...", spinner="dots"):
                result = self.system.query(question, self.ticker)

            # Display
            quality_emoji = {"correct": "✅", "ambiguous": "⚠️", "incorrect": "❌"}
            console.print(f"\n[bold]Quality:[/bold] {quality_emoji[result['quality']]} {result['quality'].upper()}")
            console.print(f"[bold]Web:[/bold] {'✅' if result['used_web'] else '❌'}")
            console.print(Panel(result['answer'], title="Answer", border_style="green"))
            console.print()
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def run(self):
        self.banner()

        # Check API keys
        required = ["OPENAI_API_KEY", "NEWSAPI_API_KEY", "TAVILY_API_KEY"]
        missing = [k for k in required if not os.getenv(k)]

        if missing:
            console.print(f"[red]❌ Missing: {', '.join(missing)}[/red]")
            return

        console.print("[green]✅ API keys OK[/green]\n")

        while True:
            try:
                prompt = f"[cyan]{self.ticker or 'CRAG'}[/cyan] > "
                user_input = Prompt.ask(prompt).strip()

                if not user_input:
                    continue

                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if cmd in ["exit", "quit"]:
                    if Confirm.ask("Exit?"):
                        console.print("[green]👋 Bye![/green]")
                        break

                elif cmd == "help":
                    self.help()

                elif cmd == "setup":
                    self.setup(args)

                elif cmd == "query":
                    self.query(args)

                else:
                    # Direct question if stock loaded
                    if self.ticker:
                        self.query(user_input)
                    else:
                        console.print("[red]Unknown command. Try 'help'[/red]")

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]❌ {e}[/red]")


def main():
    cli = CRAGCLI()
    cli.run()


if __name__ == "__main__":
    main()